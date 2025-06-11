import hydra
import paddle
from paddle.distributed import fleet

from ppcfd.data import instantiate_datamodule
from ppcfd.networks import instantiate_network
from ppcfd.utils.loss import LpLoss


def export(config):
    from paddle import jit
    from paddle.static import InputSpec

    # Initialize the model
    model = instantiate_network(config)

    # load checkpoint
    assert config.checkpoint is not None, "checkpoint must be given."
    checkpoint = paddle.load(f"{config.checkpoint}.pdparams")
    if "model_state_dict" in checkpoint:
        model.set_state_dict(checkpoint["model_state_dict"])
    else:
        model.set_state_dict(checkpoint)

    class Wrapped_Model(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, data_dict):
            pred = self.model.export(data_dict)
            return pred

    wrapped_model = Wrapped_Model(model)

    input_spec = [
        {
            "centroids": InputSpec([1, None, 3], "float32"),
            "local_centroid": InputSpec([1, None, 3], "float32"),
            "pressure": InputSpec([1, None], "float32"),
            "p_mean": InputSpec([], "float32"),
            "p_std": InputSpec([], "float32"),
        }
    ]

    jit.enable_to_static(True)
    # convert model to static graph model
    static_model = jit.to_static(wrapped_model, input_spec=input_spec, full_graph=True)
    export_path = config.output_dir + "/../inference"

    # save static graph model to disk
    try:
        jit.save(static_model, export_path, skip_prune_program=True)
    except Exception as e:
        raise e
    jit.enable_to_static(False)


def inference(config, model, test_data, device, decode):
    import paddle.inference as paddle_infer

    input_keys = ["centroids", "local_centroid", "pressure", "p_mean", "p_std"]
    # setting config
    paddle_config = paddle_infer.Config(
        f"{config.infer_path}.pdmodel", f"{config.infer_path}.pdiparams"
    )
    paddle_config.enable_use_gpu(1024, 0)
    paddle_config.enable_memory_optim()
    paddle_config.switch_use_feed_fetch_ops(False)
    paddle_config.switch_ir_optim(True)
    predictor = paddle_infer.create_predictor(paddle_config)

    output_handles = {
        "y_hat": predictor.get_output_handle(predictor.get_output_names()[0]),
        "y": predictor.get_output_handle(predictor.get_output_names()[1]),
    }

    # Initialize the dataloaders
    datamodule = instantiate_datamodule(config, 1, 0, config.n_test_num)
    test_loader = datamodule.test_dataloader(
        batch_size=config.batch_size,
        num_workers=0,
        enable_dp=False,
    )

    loss_fn = LpLoss(size_average=True)
    test_loss_p, test_loss_wss, test_loss_vel = [], [], []

    def forward():
        predictor.run()
        y_hat = paddle.to_tensor(output_handles["y_hat"].copy_to_cpu())
        y = paddle.to_tensor(output_handles["y"].copy_to_cpu())
        return y_hat, y

    def eval_dict(data, loss_fn, decode):
        out_dict = {}
        y_hat, y = forward()
        if "pressure" in data:
            out_dict["pressure l2"] = loss_fn(y_hat[:, :, 0:1], y[:, :, 0:1])
            out_dict["pressure l2 decoded"] = paddle.to_tensor([1])
            out_dict["drag_pred_pressure"] = paddle.to_tensor([1])
            out_dict["drag_truth_pressure"] = paddle.to_tensor([1])
        if "vel" in data:
            out_dict["vel l2"] = loss_fn(y_hat[:, :, 0:1], y[:, :, 0:1])
            out_dict["vel l2 decoded"] = paddle.to_tensor([1])
            out_dict["drag_pred_vel"] = paddle.to_tensor([1])
            out_dict["drag_truth_vel"] = paddle.to_tensor([1])
        if "wss" in data:
            out_dict["wss l2"] = loss_fn(y_hat[:, :, 1:4], y[:, :, 1:3])
        return out_dict

    for i, data in enumerate(test_data):
        # 准备输入数据
        input_handles = {}
        for k in input_keys:
            input_handles[k] = predictor.get_input_handle(k)
            if isinstance(data[k], paddle.Tensor):
                input_handles[k].share_external_data(data[k])
            else:
                input_handles[k].copy_from_cpu(data[k])

        if config.data_module == "drivaer":
            out_dict = eval_dict(data, loss_fn, decode)
            loss_p = out_dict["pressure l2 decoded"].item()
            loss_wss = out_dict["wss l2 decoded"].item()
            loss_vel = 0
        elif config.data_module == "pointcloud":
            y_hat, y = forward()
            loss_p = loss_fn(y_hat[None, ...], y[None, ...]).item()
            loss_wss = 0
            loss_vel = 0
        test_loss_p.append(loss_p), test_loss_wss.append(
            loss_wss
        ), test_loss_vel.append(loss_vel)
        print(
            f"Case {data['file_name']}, Relative L2 error : [P], {loss_p:.4f}, [WSS], {loss_wss:.4f}, [VEL], {loss_vel:.4f}"
        )
    print(
        f"\nRelative L2 error Mean : [P],   {sum(test_loss_p)/len(test_loss_p):.4f}, [WSS], {sum(test_loss_wss)/len(test_loss_wss):.4f}, [VEL], {sum(test_loss_vel)/len(test_loss_vel):.4f}\n"
    )


@hydra.main(version_base=None, config_path="./configs", config_name="gino.yaml")
def main(config):
    device = paddle.device.get_device()
    datamodule = instantiate_datamodule(
        config, config.n_train_num, 0, config.n_test_num
    )
    train_loader = datamodule.train_dataloader(
        batch_size=config.batch_size, num_workers=0, enable_dp=config.enable_dp
    )
    test_loader = datamodule.test_dataloader(
        batch_size=config.batch_size, num_workers=0, enable_dp=config.enable_dp
    )
    model = instantiate_network(config)
    model = fleet.distributed_model(model)

    if config.mode == "export":
        export(config)
    elif config.mode == "inference":
        inference(config, model, test_loader, device, datamodule.decode)


if __name__ == "__main__":
    main()
