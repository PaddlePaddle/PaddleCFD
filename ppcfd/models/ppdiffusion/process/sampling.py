from contextlib import ExitStack
from typing import Callable
from typing import Optional

import numpy as np
import paddle
from tqdm.auto import tqdm


class Sampling:
    def __init__(self, cfg, interp_func):
        super(Sampling, self).__init__()
        self.cfg = cfg
        self.num_timesteps = cfg.SAMPLING.num_timesteps
        self.schedule = cfg.SAMPLING.schedule
        self.pred_timesteps = cfg.SAMPLING.pred_timesteps
        self.interp_func = interp_func
        self.time_encoding = cfg.SAMPLING.time_encoding
        self.sampling_type = cfg.SAMPLING.sampling_type
        self.use_cold_sampling_for_last_step = cfg.SAMPLING.use_cold_sampling_for_last_step

        addl_interp_steps_fac = cfg.SAMPLING.addl_interp_steps_fac
        interp_before_t1 = cfg.SAMPLING.interp_before_t1
        addl_interp_steps = cfg.SAMPLING.addl_interp_steps
        self.addl_diffusion_steps = self.init_addl_diff_steps(
            addl_interp_steps_fac, interp_before_t1, addl_interp_steps
        )
        self.num_timesteps += self.addl_diffusion_steps
        self.init_interp_steps_from_diff()

        self.refine_interm_preds = cfg.SAMPLING.refine_interm_preds
        if self.refine_interm_preds:
            print("Enabling refinement of intermediate predictions.")

        # which diffusion steps to take during sampling
        self.full_sampling_schedule = list(range(0, self.num_timesteps))
        sampling_schedule = cfg.SAMPLING.sampling_schedule
        self.sampling_schedule = sampling_schedule or self.full_sampling_schedule

    def init_addl_diff_steps(self, addl_interp_steps_fac, interp_before_t1, addl_interp_steps):
        # Add additional interpolation steps to the diffusion steps
        # we substract 2 because we don't want to use the interpolator in timesteps outside [1, num_timesteps-1]
        horizon = self.num_timesteps  # = self.interpolator_horizon
        assert horizon > 1, f"Error: horizon must be > 1, but got {horizon}."
        if self.schedule == "linear":
            assert addl_interp_steps == 0, "addl_interp_steps must be 0 when using linear schedule"
            self.addl_interp_steps_fac = addl_interp_steps_fac
            interp_steps = horizon - 1 if interp_before_t1 else horizon - 2
            self.di_to_ti_add = 0 if interp_before_t1 else addl_interp_steps
            return addl_interp_steps * interp_steps
        elif self.schedule == "before_t1_only":
            assert addl_interp_steps_fac == 0, "addl_interp_steps_fac must be 0 when using before_t1_only schedule"
            assert interp_before_t1, "interp_before_t1 must be True when using before_t1_only schedule"
            return addl_interp_steps
        else:
            raise ValueError(f"Invalid schedule: {self.schedule}")

    def init_interp_steps_from_diff(self):
        self.dynamical_steps = {}
        self.interp_to_diff_step = {}
        self.artificial_interp_steps = {}

        for d in range(1, self.num_timesteps):
            i_n = self.diffu_to_interp_step(d)
            self.interp_to_diff_step[i_n] = d  # reverse mapping

            if i_n.is_integer() if isinstance(i_n, float) else i_n == int(i_n):
                self.dynamical_steps[d] = int(i_n)
            else:
                self.artificial_interp_steps[d] = i_n

    def diffu_to_interp_step(self, diffusion_step):
        """
        Convert a diffusion step to an interpolation step. Add hidden steps between step 0 and step 1 by
            mapping d_N to h-1, d_N-1 to h-2, ..., d_n to 1, and d_n-1..d_1 uniformly to [0, 1).
        e.g. if h=5, then d_5 -> 4, d_4 -> 3, d_3 -> 2, d_2 -> 1, d_1 -> 0.5
            or d_6 -> 4, d_5 -> 3, d_4 -> 2, d_3 -> 1, d_2 -> 0.66, d_1 -> 0.33
            or d_7 -> 4, d_6 -> 3, d_5 -> 2, d_4 -> 1, d_3 -> 0.75, d_2 -> 0.5, d_1 -> 0.25

        Args:
            diffusion_step (int): the diffusion step (in [1, num_timesteps-1])
        """
        if self.schedule not in ["linear", "before_t1_only"]:
            raise ValueError(f"schedule=``{self.schedule}`` not supported.")

        is_tensor = isinstance(diffusion_step, paddle.Tensor)
        is_scalar = isinstance(diffusion_step, (int, np.integer))

        if is_tensor:
            invalid_mask = (diffusion_step < 0) | (diffusion_step > self.num_timesteps - 1)
            if invalid_mask.any():
                raise ValueError(
                    f"diffusion_step must be in [1, num_timesteps-1]=[{1}, {self.num_timesteps - 1}], but got {diffusion_step}"
                )
        elif is_scalar:
            if not 0 <= diffusion_step <= self.num_timesteps - 1:
                raise ValueError(
                    f"diffusion_step must be in [1, num_timesteps-1]=[{1}, {self.num_timesteps - 1}], but got {diffusion_step}"
                )
        else:
            raise TypeError(f"Unsupported type: {type(diffusion_step)}")

        if self.schedule == "linear":
            return (diffusion_step + self.di_to_ti_add) / (self.addl_interp_steps_fac + 1)

        threshold = self.addl_diffusion_steps + 1
        if is_tensor:
            return paddle.where(
                diffusion_step >= threshold,
                (diffusion_step - self.addl_diffusion_steps).astype("float32"),
                diffusion_step / (self.addl_diffusion_steps + 1),
            )
        else:
            return (
                (diffusion_step - self.addl_diffusion_steps)
                if diffusion_step >= threshold
                else diffusion_step / (self.addl_diffusion_steps + 1)
            )

    def update_handles(self, ipol_handles=None, do_enable=True):
        self.ipol_handles = ipol_handles if ipol_handles is not None else [self]
        self.do_enable = do_enable

    def q_sample(
        self,
        x0: paddle.Tensor,
        x_end: paddle.Tensor,
        time: Optional[paddle.Tensor] = None,
        interp_time: Optional[paddle.Tensor] = None,
        is_artificial_step: bool = True,
        **kwargs,
    ) -> paddle.Tensor:
        """
        Sampling with the interpolator model.
        Just remember that x_end here refers to t=0 (the initial conditions)
        and x_0 (terminology of diffusion models) refers to t=T, i.e. the last timestep
        """
        assert time is None or interp_time is None, "Error: either time or interp_time must be None."
        time = interp_time if time is None else self.diffu_to_interp_step(time)

        with ExitStack() as stack:
            [stack.enter_context(ipol.dropout_controller(self.do_enable)) for ipol in self.ipol_handles]
            x_ti = self.interp_func(init_cond=x_end, x_last=x0, time=time, **kwargs)
        return x_ti

    def sample_loop(
        self,
        init_cond,
        sampling_fn: Callable,
        num_input_channels: int,
        static_cond: Optional[paddle.Tensor] = None,
        num_predictions: int = None,
        **kwargs,
    ):
        assert len(init_cond.shape) == 4, f"Invalid condition shape: {init_cond.shape} (should be 4D)"
        sc_kw = {"static_cond": static_cond}
        x_s = init_cond[:, -num_input_channels:]
        batch_size = init_cond.shape[0]

        intermediates = {}
        dynamics_pred_step = 0
        last_i_next_n = self.sampling_schedule[-1] + 1
        time_steps = zip(
            self.sampling_schedule,
            self.sampling_schedule[1:] + [last_i_next_n],
            self.sampling_schedule[2:] + [last_i_next_n, last_i_next_n],
        )

        for s, s_next, s_nnext in tqdm(time_steps, desc="Sampling", total=len(self.sampling_schedule), leave=False):
            is_last_step = s == self.num_timesteps - 1

            step_s = paddle.full([batch_size], s, dtype="float32")
            x0_hat = sampling_fn(x_t=x_s, condition=init_cond, time=step_s, is_sampling=True, **sc_kw)

            time_i_n = self.diffu_to_interp_step(s_next) if not is_last_step else float("inf")
            is_dynamics_pred = isinstance(time_i_n, int) or is_last_step

            q_sample_kwargs = {
                "x0": x0_hat,
                "x_end": init_cond,
                "is_artificial_step": not is_dynamics_pred,
            }
            if s_next <= self.num_timesteps - 1:
                step_s_next = paddle.full([batch_size], s_next, dtype="float32")
                x_interp_s_next = self.q_sample(**q_sample_kwargs, time=step_s_next, **sc_kw)
            else:
                x_interp_s_next = x0_hat

            if self.sampling_type == "cold":
                if is_last_step and not self.use_cold_sampling_for_last_step:
                    x_s = x0_hat
                else:
                    # D(x_s, s)
                    x_interp_s = self.q_sample(**q_sample_kwargs, time=step_s, **sc_kw) if s > 0 else x_s
                    # for s = 0, we have x_s_degraded = x_s, so we just directly return x_s_degraded_next
                    x_s += x_interp_s_next - x_interp_s
            elif self.sampling_type == "naive":
                x_s = x_interpolated_s_next
            else:
                raise ValueError(f"Error: sampling type {self.sampling_type} is not support now.")

            dynamics_pred_step = int(time_i_n) if s < self.num_timesteps - 1 else dynamics_pred_step + 1
            if is_dynamics_pred:
                intermediates[f"t{dynamics_pred_step}_preds"] = x_s

        if self.refine_interm_preds:
            # Use last prediction of x0 for final prediction of intermediate steps (not the last timestep!)
            q_sample_kwargs["x0"] = x0_hat
            q_sample_kwargs["is_artificial_step"] = False
            dynamical_steps = self.pred_timesteps or list(self.dynamical_steps.values())
            dynamical_steps = [i for i in dynamical_steps if i < self.num_timesteps]
            for i_n in dynamical_steps:
                i_n_time_tensor = paddle.full([batch_size], i_n, dtype="float32")
                i_n_for_str = int(i_n) if float(i_n).is_integer() else i_n
                assert (
                    not (i_n % 1 == 0) or f"t{i_n_for_str}_preds" in intermediates
                ), f"t{i_n_for_str}_preds not in intermediates"
                intermediates[f"t{i_n_for_str}_preds"] = self.q_sample(
                    **q_sample_kwargs, time=None, interp_time=i_n_time_tensor, **sc_kw
                )

        if last_i_next_n < self.num_timesteps:
            return x_s, intermediates, x_interp_s_next
        return x0_hat, intermediates, x_s

    @paddle.no_grad()
    def sample(self, init_cond, num_samples=1, **kwargs):
        x_0, intermediates, x_s = self.sample_loop(init_cond, **kwargs)
        return intermediates
