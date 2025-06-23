import hydra
import paddle
from paddle import jit
from paddle.static import InputSpec
from pathlib import Path
from ppcfd.utils.loss import LpLoss
import ppcfd.utils.parallel as parallel

def export(config, model):
    assert config.checkpoint is not None, "checkpoint must be given."
    checkpoint = paddle.load(f"{config.checkpoint}.pdparams")
    if "model_state_dict" in checkpoint:
        model.set_state_dict(checkpoint["model_state_dict"])
    else:
        model.set_state_dict(checkpoint)
    input_spec = [
        InputSpec([1, None, 3], "float32")
    ]
    static_model = jit.to_static(model, input_spec=input_spec)
    export_path = Path(config.output_dir) / Path(config.checkpoint).name

    jit.save(static_model, export_path.as_posix())
    print(f"Exported model to : {config.output_dir}")


def inference(config, model, datamodule):
    infer_path = Path(config.checkpoint)
    print("datamodule", datamodule)
    print("loading checkpoint from : ", infer_path)
    
    model = paddle.jit.load(infer_path.as_posix())
    model.eval()
    loss_fn = LpLoss(size_average=True)
    test_dataloader = datamodule.test_dataloader(batch_size=config.batch_size, num_workers=config.num_workers)
    targets_map = {k:i for i, k in enumerate(datamodule.test_data.targets_key)}

    mean_std_dict = datamodule.test_data.mean_std_dict
    mean = mean_std_dict["p_mean"]
    std = mean_std_dict["p_std"]
    index = config.out_keys.index("pressure")
    n = config.out_channels[index]

    model, _ = parallel.setup_module(config, model)
    test_dataloader = parallel.setup_dataloaders(config, test_dataloader)

    def denormalize(outputs, channels=0, eps=1e-6):
        return outputs[..., channels : channels + n] * (std + eps) + mean

    for _, data in enumerate(test_dataloader):
        if config.data_module._target_ == "ppcfd.data.DrivAerDataModule":
            y = model(data["inputs"])
        elif config.data_module._target_ == "ppcfd.data.PointCloudDataModule":
            y = model(data["inputs"])
            p_pred = denormalize(y)
            p_true = data["targets"][targets_map["pressure"]]
        print(f"l2 error: {loss_fn(p_true, p_pred).item():.3f}")


@hydra.main(version_base=None, config_path="./configs", config_name="gino.yaml")
def main(config):
    datamodule = hydra.utils.instantiate(config.data_module)
    model = hydra.utils.instantiate(config.model)

    if config.mode == "export":
        export(config, model)
    elif config.mode == "inference":
        inference(config, model, datamodule)


if __name__ == "__main__":
    main()
