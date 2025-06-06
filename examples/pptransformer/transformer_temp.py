import logging
import os
import re
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import hydra
import numpy as np
import paddle
import pandas as pd
import tensorboardX
from paddle.io import BatchSampler

import ppcfd.utils.op as op
import ppcfd.utils.parallel as parallel
from ppcfd.utils.loss import LpLoss
from ppcfd.utils.metric import R2Score

log = logging.getLogger(__name__)


def set_seed(seed: int = 0):
    paddle.seed(seed)
    np.random.seed(seed)


@dataclass
class AeroDynamicCoefficients:
    c_p_pred: op.Tensor = op.to_tensor(1.0)
    c_f_pred: op.Tensor = op.to_tensor(1.0)
    c_d_pred: op.Tensor = op.to_tensor(1.0)
    c_l_pred: op.Tensor = op.to_tensor(1.0)
    c_p_true: op.Tensor = op.to_tensor(1.0)
    c_f_true: op.Tensor = op.to_tensor(1.0)
    c_l_true: op.Tensor = op.to_tensor(1.0)
    c_d_true: op.Tensor = op.to_tensor(1.0)
    c_p_mre: op.Tensor = op.to_tensor(0.0)
    c_f_mre: op.Tensor = op.to_tensor(0.0)
    c_d_mre: op.Tensor = op.to_tensor(0.0)
    c_l_mre: op.Tensor = op.to_tensor(0.0)
    cd_starccm: op.Tensor = op.to_tensor(1.0)
    reference_area: op.Tensor = op.to_tensor(-1.0)


@dataclass
class AeroDynamicLoss:
    l2_p: list = field(default_factory=list)
    mse_cd: list = field(default_factory=list)
    mre_cp: list = field(default_factory=list)


@dataclass
class AeroDynamicMetrics:
    csv_title: list = field(
        default_factory=lambda: [
            "file_name",
            "cp pred",
            "cf pred",
            "cp true",
            "cf true",
            "cd starccm+",
            "frontal area",
        ]
    )
    test_l2_loss_list: list = field(default_factory=lambda: [0.0])
    test_mse_cd_loss_list: list = field(default_factory=list)
    test_l2_loss_list_p: list = field(default_factory=list)
    test_l2_loss_list_wss: list = field(default_factory=list)
    test_l2_loss_list_vel: list = field(default_factory=list)
    test_mse_loss_list_p: list = field(default_factory=list)
    test_mse_loss_list_wss: list = field(default_factory=list)
    test_mse_loss_list_vel: list = field(default_factory=list)
    test_cp_mre_list: list = field(default_factory=list)
    test_cf_mre_list: list = field(default_factory=list)
    test_cd_mre_list: list = field(default_factory=list)
    test_cl_mre_list: list = field(default_factory=list)
    cp_r2_score: float = 0.0
    cf_r2_score: float = 0.0
    cl_r2_score: float = 0.0
    cd_r2_score: float = 0.0


@dataclass
class AeroDynamicPhysicsField:
    u: op.Tensor = op.to_tensor(0.0)
    v: op.Tensor = op.to_tensor(0.0)
    w: op.Tensor = op.to_tensor(0.0)
    p: op.Tensor = op.to_tensor(0.0)
    wss: op.Tensor = op.to_tensor(0.0)
    wss_x: op.Tensor = op.to_tensor(0.0)
    cd: op.Tensor = op.to_tensor(0.0)


@dataclass
class StructuralCoefficients:
    mass: op.Tensor = op.to_tensor(0.0)
    safety_factor: op.Tensor = op.to_tensor(0.0)
    max_displacement: op.Tensor = op.to_tensor(0.0)
    contact_pressure: op.Tensor = op.to_tensor(0.0)
    max_mises_stress: op.Tensor = op.to_tensor(0.0)
    max_shear_stress: op.Tensor = op.to_tensor(0.0)
    total_strain_energy: op.Tensor = op.to_tensor(0.0)
    max_principal_stress: op.Tensor = op.to_tensor(0.0)
    max_von_mises_strain: op.Tensor = op.to_tensor(0.0)


@dataclass
class StructuralMetrics:
    csv_title: list = field(default_factory=lambda: ["file_name", "mean L-2 error"])
    l2: list = field(default_factory=list)


@dataclass
class StructuralLoss:
    l2: list = field(default_factory=list)


class Car_Loss:
    def __init__(self, config):
        self.config = config
        self.l2_mean_loss = LpLoss(p=2)
        self.cx_list = []
        self.mse_loss = paddle.nn.MSELoss()
        self.metric = AeroDynamicMetrics()

    def __call__(
        self, inputs, outputs, targets, others, loss_fn, loss_cd_fn, cal_metric=False
    ):
        config = self.config
        l2_loss, mse_cd_loss, c_p_mre, loss_p, loss_wss, loss_vel = [
            op.to_tensor([0.0])
        ] * 6
        [output, label], [pred, true] = self.denormalize(
            outputs, targets, others, config.mode
        )
        cx = self.calculate_coefficient(
            inputs,
            targets,
            others,
            pred=pred,
            true=true,
            mass_density=config.mass_density,
            flow_speed=config.flow_speed,
        )
        targets["coefficient"] = cx

        if self.config.mode == "inference":
            return cx, pred

        if "pressure" in config.out_keys:
            loss_p = loss_fn(pred.p, true.p)

        if "wss" in config.out_keys:
            loss_wss = loss_fn(pred.wss, true.wss)

        if "vel" in config.out_keys:
            loss_vel = loss_fn(pred.u, true.u)

        l2_loss = loss_fn(output, label)
        if config.cd_finetune is True:
            mse_cd_loss = loss_cd_fn(cx.c_p_pred, cx.c_p_true)
        else:
            mse_cd_loss = op.to_tensor([0.0])

        return_list = [
            l2_loss,
            mse_cd_loss,
            loss_p,
            loss_wss,
            loss_vel,
        ]
        if cal_metric:
            metrics_updated = self.update(return_list, cx)
            return metrics_updated
        else:
            return return_list

    def update(self, loss_list, cx):
        m = self.metric
        m.test_l2_loss_list.append(loss_list[0].item())
        m.test_mse_cd_loss_list.append(loss_list[1].item())
        m.test_l2_loss_list_p.append(loss_list[2].item())
        m.test_l2_loss_list_wss.append(loss_list[3].item())
        m.test_l2_loss_list_vel.append(loss_list[4].item())
        m.test_cp_mre_list.append(cx.c_p_mre.item())
        m.test_cf_mre_list.append(cx.c_f_mre.item())
        m.test_cd_mre_list.append(cx.c_d_mre.item())
        m.test_cl_mre_list.append(cx.c_l_mre.item())
        return m

    def integral_over_cells(
        self,
        reference_area,
        surface_normals,
        areas,
        mass_density,
        flow_speed,
        x_direction=1,
    ):
        flow_normals = op.zeros(surface_normals.shape)
        flow_normals[..., 0] = x_direction
        const = 2.0 / (mass_density * flow_speed**2 * reference_area)
        const = op.to_tensor(const)
        const = const.reshape([-1, 1, 1])
        direction = op.sum(surface_normals * flow_normals, axis=-1, keepdim=True)
        c_p = const * direction * areas
        c_f = (const * flow_normals * areas)[..., 0:1]
        return c_p, c_f

    def calculate_coefficient(
        self,
        inputs,
        targets,
        others,
        pred,
        true,
        mass_density,
        flow_speed,
        x_direction=1,
    ):
        cx = AeroDynamicCoefficients()
        if "Cd" in self.config.out_keys:
            cx.c_d_pred = pred.cd
            cx.c_d_true = true.cd
            cx.c_d_mre = abs(cx.c_d_pred - cx.c_d_true) / abs(cx.c_d_true)
            return cx
        else:
            cx.cd_starccm = others.get("Cd", 1.0)
            cx.reference_area = others["reference_area"]
        if "pressure" in self.config.out_keys or "wss" in self.config.out_keys:
            # 2. Prepare Discreted Integral over Car Surface
            cp, cf = self.integral_over_cells(
                others["reference_area"],
                targets["normal"],
                targets["areas"],
                mass_density,
                flow_speed,
                x_direction,
            )

            # 3. Calculate coefficient and MRE
            if "pressure" in self.config.out_keys:
                cx.c_p_pred = op.sum(cp * pred.p, axis=[1, 2])
                cx.c_p_true = op.sum(cp * true.p, axis=[1, 2])
                cx.c_p_mre = abs(cx.c_p_pred - cx.c_p_true) / abs(cx.c_p_true)

            if "wss" in self.config.out_keys:
                cx.c_f_pred = op.sum(cf * pred.wss_x, axis=[1, 2])
                cx.c_f_true = op.sum(cf * true.wss_x, axis=[1, 2])
                cx.c_f_mre = abs(cx.c_f_pred - cx.c_f_true) / abs(cx.c_f_true)

            if {"pressure", "wss"}.issubset(self.config.out_keys):
                cx.c_d_pred = cx.c_p_pred + cx.c_f_pred
                cx.c_d_true = cx.c_p_true + cx.c_f_true
                cx.c_d_mre = abs(cx.c_d_pred - cx.c_d_true) / abs(cx.c_d_true)
                cx.c_l_pred = cx.c_p_pred  # tofix
                cx.c_l_true = cx.c_p_true  # tofix
                cx.c_l_mre = abs(
                    op.to_tensor([1e-5] * inputs["centroids"].shape[0])
                ) / abs(cx.c_l_true)
        return cx

    def denormalize(self, outputs, targets, others, mode, eps=1e-6):
        config = self.config
        channels = 0
        true = AeroDynamicPhysicsField()
        pred = AeroDynamicPhysicsField()
        label_list, pred_list = [], []
        (
            p_true,
            p_pred,
            wss_true,
            wss_pred,
            wss_x_true,
            wss_x_pred,
            vel_true,
            vel_pred,
            cd_true,
            cd_pred,
        ) = [None] * 10
        assert len(config.out_keys) != 0, "config.out_keys must be not empty"
        if "pressure" in config.out_keys:
            if mode in ["test", "train"]:
                p_true = targets["pressure"].unsqueeze(axis=2)
                label_list.append(p_true)
                true.p = p_true
            mean = others["p_mean"]
            std = others["p_std"]
            index = config.out_keys.index("pressure")
            n = config.out_channels[index]
            p_pred = outputs[..., channels : channels + n] * (std + eps) + mean
            pred_list.append(p_pred)
            pred.p = p_pred
            channels += n
        if "wss" in config.out_keys:
            if mode in ["test", "train"]:
                wss_true = targets["wss"]
                label_list.append(wss_true)
                wss_x_true = wss_true[..., 0:1]
                true.wss = wss_true
                true.wss_x = wss_x_true
            mean = others["wss_mean"]
            std = others["wss_std"]
            index = config.out_keys.index("wss")
            n = config.out_channels[index]
            wss_pred = outputs[..., channels : channels + n] * (std + eps) + mean
            wss_x_pred = wss_pred[..., 0:1]
            pred_list.append(wss_pred)
            pred.wss = wss_pred
            pred.wss_x = wss_x_pred
            channels += n
        if "vel" in config.out_keys:
            if mode in ["test", "train"]:
                vel_true = targets["vel"].unsqueeze(axis=2)
                label_list.append(vel_true)
                true.u = vel_true

            mean = others.get("v_mean", [1.0])[0]
            std = others.get("v_std", [0.0])[0]
            index = config.out_keys.index("vel")
            n = config.out_channels[index]
            vel_pred = outputs[..., channels : channels + n] * mean + std
            pred_list.append(vel_pred)
            pred.u = vel_pred
            channels += n
        if "Cd" in config.out_keys:
            index = config.out_keys.index("Cd")
            n = config.out_channels[index]
            cd_pred = outputs[..., channels : channels + n]
            cd_true = others["Cd"]
            pred_list.append(cd_pred)
            label_list.append(cd_true)
            true.cd = cd_true
            pred.cd = cd_pred

        pred_list = op.concat(pred_list, axis=-1)
        if mode in ["test", "train"]:
            label_list = op.concat(label_list, axis=-1)
        return [pred_list, label_list], [pred, true]


class Structural_Loss:
    def __init__(self, config):
        self.config = config
        self.structural_loss = StructuralLoss()
        self.structural_metric = StructuralMetrics()

    def __call__(
        self, inputs, outputs, targets, others, loss_fn, loss_cd_fn, cal_metric=False
    ):
        targets["coefficient"] = StructuralCoefficients()
        loss_list = []
        for k in self.config.out_keys:
            _targets = targets[k]
            outputs = abs(outputs)  # tofix
            l2_loss = loss_fn(outputs, _targets)
            loss_list.append(l2_loss)
            if cal_metric:
                self.structural_metric.l2.append(l2_loss.item())
            else:
                self.structural_loss.l2.append(l2_loss.item())
            output_dir = Path(self.config.output_dir) / "test_case"
            output_dir.mkdir(parents=True, exist_ok=True)
            if k == "stress":
                output_df = pd.DataFrame(
                    {
                        "x": inputs["centroids"][0, :, 0].numpy(),
                        "y": inputs["centroids"][0, :, 1].numpy(),
                        "z": inputs["centroids"][0, :, 2].numpy(),
                        k: targets[k][0, :, 0].numpy(),
                        "output": outputs[0, :, 0].numpy(),  # 假设output是可迭代对象
                    }
                )
            elif k == "natural_frequency":
                output_df = pd.DataFrame(
                    {
                        k: targets[k][0, :, 0].numpy(),
                        "output": outputs[0, :, 0].numpy(),  # 假设output是可迭代对象
                    }
                )
            else:
                raise NotImplementedError

            # 保存为CSV
            output_df.to_csv(
                output_dir / f"{others['file_name']}_test_train.csv",
                index=False,  # 不保存行索引
            )
        if cal_metric:
            return self.structural_metric
        else:
            return loss_list


class Loss_logger:
    def __init__(self, output_dir, mode, simulation_type):
        self.output_dir = Path(output_dir)
        self.mode = mode
        if "Structural" == simulation_type:
            self.metric = StructuralMetrics()
            self.loss_list = StructuralLoss()
        elif "AeroDynamic" == simulation_type:
            self.metric = AeroDynamicMetrics()
            self.loss_list = AeroDynamicLoss()
        else:
            raise ValueError("loss_fn must be StructuralMetrics or AeroDynamicMetrics")

        tensorboard = tensorboardX.SummaryWriter(
            os.path.join(output_dir, "tensorboard")
        )
        log.info(f"Working directory : {os.getcwd()}")
        log.info(f"Output directory  : {output_dir}")
        self.tensorboard = tensorboard

        self.csv_list = [self.metric.csv_title]
        self.cx_test_list = []

    def record_train_loss(self, input_list):
        loss = self.loss_list
        input_list = [op.mean(x).item() if x is op.Tensor else x for x in input_list]
        if isinstance(loss, AeroDynamicLoss):
            cx = input_list[2]
            loss.l2_p.append(input_list[0])
            loss.mse_cd.append(input_list[1])
            loss.mre_cp.append(cx.c_p_mre)
        elif isinstance(loss, StructuralLoss):
            loss.l2.append(input_list[0])
        else:
            raise ValueError("loss_fn must be StructuralLoss or AeroDynamicLoss")

    def record_metric_report(self):
        m = self.metric
        df = pd.DataFrame(self.csv_list[1:], columns=self.csv_list[0])
        df.to_csv(self.output_dir / "test.csv", mode="w", index=False)
        if self.mode == "test":
            if isinstance(m, AeroDynamicMetrics):
                outputs = {
                    # "Cp": op.to_tensor([cx.c_p_pred for cx in self.cx_test_list]),
                    # "cf": op.to_tensor([cx.c_f_pred for cx in self.cx_test_list]),
                    "Cd": op.to_tensor([cx.c_d_pred for cx in self.cx_test_list]),
                }

                targets = {
                    # "Cp": op.to_tensor([cx.c_p_true for cx in self.cx_test_list]),
                    # "cf": op.to_tensor([cx.c_f_true for cx in self.cx_test_list]),
                    "Cd": op.to_tensor([cx.c_d_true for cx in self.cx_test_list]),
                }
                r2_metric = R2Score()
                r2_metric_dict = r2_metric(outputs, targets)
                # m.cp_r2_score=r2_metric_dict["cp"]
                # m.cf_r2_score=r2_metric_dict["cf"]
                m.cd_r2_score = r2_metric_dict["Cd"]
                # m.cp_r2_score=r2_metric_dict["Cp"]
                log.info(
                    f"MRE Error Mean: [Cp], {np.mean(m.test_cp_mre_list)*100:.2f}%, [Cf], {np.mean(m.test_cf_mre_list)*100:.2f}%, [Cd], {np.mean(m.test_cd_mre_list)*100:.2f}%, [Cl], {np.mean(m.test_cl_mre_list)*100:.2f}% \tRelative L2 Error Mean: [P], {np.mean(m.test_l2_loss_list_p):.4f}, [WSS], {np.mean(m.test_l2_loss_list_wss):.4f}, [VEL], {np.mean(m.test_l2_loss_list_vel):.4f}, R2 Score: [Cd], {m.cd_r2_score:.2f}"
                )
            elif isinstance(m, StructuralMetrics):
                log.info(f"Mean Relative L-2 Error [Stress]: {np.mean(m.l2):.2f}")

    def record_metric(self, file_name, cx, metric):
        self.cx_test_list.append(cx)
        self.metric = metric
        if isinstance(metric, AeroDynamicMetrics):
            log.info(
                f"Case {file_name}\t MRE Error : [Cp], {cx.c_p_mre.item()*100:.2f}%, [Cf], {cx.c_f_mre.item()*100:.2f}%, [Cd], {cx.c_d_mre.item()*100:.2f}%, [Cl], {cx.c_l_mre.item()*100:.2f}% \tRelative L2 Error : [P], {metric.test_l2_loss_list_p[-1]:.4f}, [WSS], {metric.test_l2_loss_list_wss[-1]:.4f}, [VEL], {metric.test_l2_loss_list_vel[-1]:.4f}"
            )
            self.csv_list.append(
                [
                    file_name,
                    cx.c_p_pred.item(),
                    cx.c_f_pred.item(),
                    cx.c_p_true.item(),
                    cx.c_f_true.item(),
                    cx.cd_starccm.item(),
                    cx.reference_area.item(),
                ]
            )
        elif isinstance(metric, StructuralMetrics):
            if self.mode == "test":
                log.info(
                    f"Case {file_name}\t, Mean L-2 Relative Error [Stress]: {metric.l2[-1]:.2f}"
                )
            self.csv_list.append([file_name, metric.l2[-1]])
        else:
            raise ValueError("loss_fn must be StructuralMetrics or AeroDynamicMetrics")

    def record_tensorboard(
        self,
        ep,
        time_cost,
        lr,
    ):
        loss = self.loss_list
        m = self.metric
        if isinstance(loss, AeroDynamicLoss) and isinstance(m, AeroDynamicMetrics):
            self.tensorboard.add_scalar("Train_L2", np.mean(loss.l2_p), ep)
            self.tensorboard.add_scalar("Train_Cd_MSE", np.mean(loss.mse_cd), ep)
            self.tensorboard.add_scalar("Test_L2", np.mean(m.test_l2_loss_list), ep)
            self.tensorboard.add_scalar(
                "Test_Cd_MSE", np.mean(m.test_mse_cd_loss_list), ep
            )
            log.info(
                f"Epoch {ep}  Times {(time_cost):.2f}s, lr:{lr:.1e}, [Tain] L2 :{np.mean(loss.l2_p):.4f},  MSE : [Cd] {np.mean(loss.mse_cd):.1e},  MRE: [Cp]{100*np.mean(loss.mre_cp):.2f}% ,  [Test] L2 :{np.mean(m.test_l2_loss_list):.4f},  MSE: [Cd] {np.mean(m.test_mse_cd_loss_list):.2f}, MRE: [Cp] {100*np.mean(m.test_cp_mre_list):.2f}%, [Cf] {100*np.mean(m.test_cf_mre_list):.2f}%, [Cl] {100*np.mean(m.test_cl_mre_list):.2f}%"
            )
        elif isinstance(m, StructuralMetrics) and isinstance(loss, StructuralLoss):
            self.tensorboard.add_scalar("Train_L2", np.mean(loss.l2), ep)
            self.tensorboard.add_scalar("Test_L2", np.mean(m.l2), ep)
            log.info(
                f"Epoch {ep}  Times {(time_cost):.2f}s, lr:{lr:.1e}, [Tain] Mean Relative L2 loss:{np.mean(m.l2):.4f}   [Test] Mean Relative L2 loss:{np.mean(loss.l2):.4f}"
            )
        else:
            raise ValueError("loss_fn must be StructuralMetrics or AeroDynamicMetrics")


def load_checkpoint(config, model, optimizer=None):
    assert config.checkpoint is not None, "checkpoint must be given."
    checkpoint_path = Path(config.checkpoint)
    ep_start = int((re.search(r"_(\d+)", checkpoint_path.name)).group(1))
    ckpt_path = f"{config.checkpoint}.pdparams"
    opt_path = f"{config.checkpoint}.pdopt"
    if os.path.isdir(ckpt_path):
        op.load_state_dict(optimizer.state_dict(), opt_path)
    else:
        pdparams = op.load(ckpt_path)
        ep_start = int((re.search(r"_(\d+)", config.checkpoint)).group(1))
        if optimizer is not None and os.path.exists(opt_path):
            pdopt = op.load(opt_path)
            optimizer.set_state_dict(pdopt)
        if "model_state_dict" in pdparams:
            model.set_state_dict(pdparams["model_state_dict"])
        else:
            model.set_state_dict(pdparams)
    return ep_start


@paddle.no_grad()
def test(config, model, test_dataloader, loss_logger, ep=None):
    if config.mode == "test":
        load_checkpoint(config, model)
        full_batch = True
    else:
        full_batch = ((ep + 1) % config.val_freq == 0) or (
            (ep + 1) == config.num_epochs
        )
    model.eval()
    loss_cd_fn = op.mse_fn()
    loss_fn = LpLoss(size_average=True)
    if config.simulation_type == "AeroDynamic":
        simulation_loss = Car_Loss(config)
    elif config.simulation_type == "Structural":
        simulation_loss = Structural_Loss(config)
    else:
        raise ValueError(f"Invalid simulation type. {config.simulation_type}")

    # parallel
    config.enable_dp = False  # length different for each batch
    model, _ = parallel.setup_module(config, model)
    data_to_dict = test_dataloader.dataset.data_to_dict
    test_dataloader = parallel.setup_dataloaders(config, test_dataloader)
    t0 = time.time()
    for i, data in enumerate(test_dataloader):
        with paddle.no_grad():
            outputs = model(data["inputs"])
        inputs, targets, others = data_to_dict(data)
        metric = simulation_loss(
            inputs, outputs, targets, others, loss_fn, loss_cd_fn, cal_metric=True
        )
        loss_logger.record_metric(
            others.get("file_name", f"Test Case {i}"),
            targets.get("coefficient", AeroDynamicCoefficients()),
            metric,
        )

    loss_logger.record_metric_report()
    if config.mode == "test":
        log.info(
            f"Test finished. time: {float(time.time() - t0):.3f} seconds, max gpu memory = {paddle.device.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )


def train(config, model, datamodule, eval_dataloader, loss_logger):
    """
    使用PaddlePaddle框架训练模型

    Args:
        config (dict): 配置信息，包含学习率(lr)和训练轮数(num_epochs)等
        model (paddle.nn.Layer): 待训练的模型

    Returns:
        None
    """
    model.train()

    # 损失函数
    loss_fn = LpLoss(size_average=True)
    loss_cd_fn = op.mse_fn()
    car_loss = Car_Loss(config)
    structural_loss = Structural_Loss(config)

    # 创建优化器
    optimizer = op.adamw_fn(
        parameters=model.parameters(), learning_rate=config.lr, weight_decay=1e-6
    )

    # 断点续训
    if config.checkpoint is not None:
        log.info(f"loading checkpoint from: {config.checkpoint}")
        ep_start = load_checkpoint(config, model, optimizer)
    else:
        ep_start = 0
    # 学习率调度器
    optimizer, scheduler = op.lr_schedular_fn(
        scheduler_name=config.lr_schedular,
        learning_rate=optimizer.get_lr(),
        T_max=config.num_epochs,
        optimizer=optimizer,
    )

    train_sampler = BatchSampler(
        datamodule.train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader = datamodule.train_dataloader(
        num_workers=config.num_workers,
        batch_sampler=train_sampler,
    )
    train_dataloader = parallel.setup_dataloaders(config, train_dataloader, datamodule)
    model, optimizer = parallel.setup_module(config, model, optimizer)

    data_to_dict = train_dataloader.dataset.data_to_dict
    log.info(
        f"iters per epochs = {len(train_dataloader)}, num_train = {config.data_module.n_train_num}, total bacth size = {config.batch_size * paddle.distributed.get_world_size()}"
    )
    t0 = time.time()
    for ep in range(ep_start, config.num_epochs):
        t1 = time.time()
        # 训练循环
        for n_iter, data in enumerate(train_dataloader):
            outputs = model(data["inputs"])
            inputs, targets, others = data_to_dict(data)
            # 模型前向传播
            if config.simulation_type == "AeroDynamic":
                loss_list = car_loss(
                    inputs, outputs, targets, others, loss_fn, loss_cd_fn
                )
                l2_loss = loss_list[0]
                mse_cd_loss = loss_list[1]

                if config.cd_finetune is True:
                    train_loss = l2_loss + config.cd_loss_weight * mse_cd_loss
                else:
                    train_loss = l2_loss
                loss_logger.record_train_loss(
                    [
                        train_loss,
                        mse_cd_loss,
                        targets.get("coefficient", AeroDynamicCoefficients()),
                    ]
                )
            elif config.simulation_type == "Structural":
                loss_list = structural_loss(
                    inputs, outputs, targets, others, loss_fn, loss_cd_fn
                )
                loss_logger.record_train_loss(loss_list)
                l2_loss = loss_list[0]
                train_loss = l2_loss
            # 清除梯度
            optimizer.clear_grad()
            # 反向传播
            train_loss.backward()
            # 更新模型参数
            optimizer.step()
        # 更新学习率
        if config.lr_schedular is not None:
            scheduler.step()
        # 测试循环
        test(config, model, eval_dataloader, loss_logger, ep)
        model.train()
        # 打印训练信息
        loss_logger.record_tensorboard(
            ep,
            (time.time() - t1),
            optimizer.get_lr(),
        )

        # Save the weights
        if ((ep + 1) % 50 == 0) or ((ep + 1) == config.num_epochs):
            model_name = f"{config.output_dir}/{config.model_name}_{ep}"
            if config.enable_mp is True or config.enable_pp is True:
                op.save_state_dict(model.state_dict(), f"{model_name}.pdparams")
                state = optimizer.state_dict()
                if config.lr_schedular is not None:
                    state.pop("LR_Scheduler")
                op.save_state_dict(state, f"{model_name}.pdopt")
            else:
                op.save(model.state_dict(), f"{model_name}.pdparams")
                op.save(optimizer.state_dict(), f"{model_name}.pdopt")
    log.info(
        f"Training finished. time: {float(time.time() - t0)/3600:.3f} hours, max gpu memory = {paddle.device.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )


@hydra.main(version_base=None, config_path="./configs", config_name="transolver.yaml")
def main(config):
    """
    主函数，用于训练和测试模型

    Args:
        config (dict): 包含模型配置信息的字典

    Returns:
        None
    """

    # 初始化损失记录器
    loss_logger = Loss_logger(config.output_dir, config.mode, config.simulation_type)

    # 设置随机种子
    set_seed(config.seed)

    # 数据生成
    datamodule = hydra.utils.instantiate(config.data_module)

    # 模型构建
    model = hydra.utils.instantiate(config.model)

    # 模型训练
    test_dataloader = datamodule.test_dataloader(
        batch_size=config.batch_size, num_workers=config.num_workers
    )
    # eval_dataloader = datamodule.eval_dataloader(
    #     batch_size=config.batch_size, num_workers=config.num_workers
    # )
    if config.mode == "train":
        train(config, model, datamodule, test_dataloader, loss_logger)
    # 模型测试
    elif config.mode == "test":
        # 创建测试数据加载器
        model.eval()
        test(config, model, test_dataloader, loss_logger)


if __name__ == "__main__":
    main()
