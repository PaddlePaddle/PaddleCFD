from __future__ import annotations

import json
import pathlib
from collections import namedtuple

import numpy as np
from einops import rearrange


Trajectory = namedtuple(
    "Trajectory",
    [
        "name",
        "features",
        "dp_dt",
        "dq_dt",
        "t",
        "trajectory_meta",
        "p_noiseless",
        "q_noiseless",
        "masses",
        "edge_index",
        "vertices",
        "fixed_mask",
        "condition",
        "static_nodes",
    ],
)


class TrajectoryDataset:
    """Returns batches of full trajectories.

    dataset[idx] -> a set of snapshots for a full trajectory

    For 'navier-stokes':
    core values of this system: solutions(x) and pressures(y)
    metadata(dict).keys()=['system', 'system_args', 'metadata', 'trajectories']
    'system' is 'navier-stokes'
    'system_args.trajectory_defs' has 'viscosity' and 'in_velocity',
        which are parameters passed to the FEM solver giving the viscosity of
        the fluid and the velocity of the incoming flow
    'metadata' has 'grid_resolution' and 'viscosity'
    'trajectories' have 'in_velocity', and 'viscosity'
    """

    def __init__(self, data_dir, subsample: int = 1, max_samples: int = None):
        super().__init__()
        data_dir = pathlib.Path(data_dir)
        self.subsample = subsample
        self.max_samples = max_samples

        with open(data_dir / "system_meta.json", "r", encoding="utf8") as meta_file:
            metadata = json.load(meta_file)

        self.system = metadata["system"]
        self.system_metadata = metadata["metadata"]
        self._trajectory_meta = metadata["trajectories"]
        self._npz_file = np.load(data_dir / "trajectories.npz")
        if self.system == "navier-stokes":
            self.h, self.w = 221, 42
            self._ndims_p = 2
            self._ndims_q = 1
        else:
            raise ValueError(f"Unknown system: {self.system}")

    def concatenate_features(self, p, q, channel_dim=-1):
        """How to concatenate any item in the dataset"""
        q = np.expand_dims(q, axis=channel_dim) if q.shape[channel_dim] != 1 and q.ndim <= 2 else q
        assert p.shape[channel_dim] == 2, f"Expected p to have 2 channels but got {p.shape}"
        assert q.shape[channel_dim] == self._ndims_q, f"Expected q to have {self._ndims_q} channel, but got {q.shape}"
        dynamics = np.concatenate([p, q], axis=channel_dim)
        return dynamics

    def get_others(self, field_keys, p, q):
        dp_dt = self._npz_file[field_keys["dpdt"]]
        dq_dt = self._npz_file[field_keys["dqdt"]]

        # Handle (possibly missing) noiseless data
        p_key = "p_noiseless" if "p_noiseless" in field_keys else "p"
        q_key = "q_noiseless" if "q_noiseless" in field_keys else "q"
        p_noiseless = self._npz_file[field_keys[p_key]]
        q_noiseless = self._npz_file[field_keys[q_key]]

        # Handle (possibly missing) masses
        masses = self._npz_file[field_keys["masses"]] if "masses" in field_keys else np.ones(p.shape[1])
        if "edge_indices" in field_keys:
            edge_index = self._npz_file[field_keys["edge_indices"]]
            if edge_index.shape[0] != 2:
                edge_index = edge_index.T
        else:
            edge_index = []
        vertices = self._npz_file[field_keys["vertices"]] if "vertices" in field_keys else []

        # Handle per-trajectory boundary masks
        fixed_mask_p = (
            np.expand_dims(self._npz_file[field_keys["fixed_mask_p"]], 0) if "fixed_mask_p" in field_keys else []
        )
        fixed_mask_q = (
            np.expand_dims(self._npz_file[field_keys["fixed_mask_q"]], 0) if "fixed_mask_q" in field_keys else []
        )
        extra_fixed_mask = (
            np.expand_dims(self._npz_file[field_keys["extra_fixed_mask"]], 0)
            if "extra_fixed_mask" in field_keys
            else []
        )
        static_nodes = (
            np.expand_dims(self._npz_file[field_keys["enumerated_fixed_mask"]], 0)
            if "enumerated_fixed_mask" in field_keys
            else []
        )

        # set parrerns
        efm_pattern = "1 (h w) c -> c h w" if extra_fixed_mask.ndim == 3 else "1 (h w) -> 1 h w"
        q_pattern = "time (h w) -> time 1 h w" if q.ndim == 2 else "time (h w) c -> time c h w"
        q_static_pattern = "(h w) -> h w" if q.ndim == 2 else "(h w) c -> c h w"

        extra_fixed_mask = rearrange(extra_fixed_mask, efm_pattern, h=self.h, w=self.w)
        dp_dt = rearrange(dp_dt, "time (h w) c -> time c h w", h=self.h, w=self.w).astype(np.float32)
        dq_dt = rearrange(dq_dt, q_pattern, h=self.h, w=self.w).astype(np.float32)
        p_noiseless = rearrange(p_noiseless, "time (h w) c -> time c h w", h=self.h, w=self.w).astype(np.float32)
        q_noiseless = rearrange(q_noiseless, q_pattern, h=self.h, w=self.w).astype(np.float32)

        masses = rearrange(masses, "(h w) -> h w", h=self.h, w=self.w)
        vertices = (
            rearrange(vertices, "(h w) c -> c h w", h=self.h, w=self.w).astype(np.float32) if len(vertices) > 0 else []
        )
        static_nodes = (
            rearrange(static_nodes.squeeze(), "(h w) -> h w", h=self.h, w=self.w) if len(static_nodes[0]) > 0 else []
        )
        fixed_mask_p = rearrange(fixed_mask_p.squeeze(), "(h w) c -> c h w", h=self.h, w=self.w)
        fixed_mask_q = rearrange(fixed_mask_q.squeeze(), q_static_pattern, h=self.h, w=self.w)
        fixed_mask = self.concatenate_features(fixed_mask_p, q=fixed_mask_q, channel_dim=0)
        return (
            extra_fixed_mask,
            dp_dt,
            dq_dt,
            p_noiseless,
            q_noiseless,
            masses,
            vertices,
            static_nodes,
            fixed_mask,
            edge_index,
        )

    def __getitem__(self, idx):
        meta = self._trajectory_meta[idx]
        name = meta["name"]
        field_keys = meta["field_keys"]

        # Load arrays
        # print([self._trajectory_meta[idx]["name"] for idx in range(self.__len__())])
        # print(f'----> Loading trajectory {name}', field_keys["p"])

        p = self._npz_file[field_keys["p"]]
        q = self._npz_file[field_keys["q"]]
        # concatenate the p, q variables into the feature dimension
        features = self.concatenate_features(p, q, channel_dim=-1)
        # reconstruct spatial dimensions, from (time, 221*42, channel) to (time, channel, 221, 42)
        features = rearrange(features, "time (h w) c -> time c h w", h=self.h, w=self.w).astype(np.float32)

        t = self._npz_file[field_keys["t"]]
        (
            extra_fixed_mask,
            dp_dt,
            dq_dt,
            p_noiseless,
            q_noiseless,
            masses,
            vertices,
            static_nodes,
            fixed_mask,
            edge_index,
        ) = self.get_others(field_keys, p, q)

        if self.subsample > 1:
            meta["time_step_size"] = meta["time_step_size"] * self.subsample
            features = features[:: self.subsample]
            dp_dt = dp_dt[:: self.subsample]
            dq_dt = dq_dt[:: self.subsample]
            p_noiseless = p_noiseless[:: self.subsample]
            q_noiseless = q_noiseless[:: self.subsample]
            t = t[:: self.subsample]
            meta["num_time_steps"] = len(t)

        # Package and return
        return Trajectory(
            name=name,
            trajectory_meta=meta,
            features=features,
            dp_dt=dp_dt,
            dq_dt=dq_dt,
            t=t,
            p_noiseless=p_noiseless,
            q_noiseless=q_noiseless,
            masses=masses,
            edge_index=edge_index,
            vertices=vertices,
            fixed_mask=fixed_mask,
            condition=extra_fixed_mask,
            static_nodes=static_nodes,
        )

    def __len__(self):
        return len(self._trajectory_meta) if self.max_samples is None else self.max_samples
