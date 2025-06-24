from .datamodule_lazy import SAEDataModule
from .inference_dataloader import SAEInferenceDataModule


def instantiate_datamodule(
    config,
    dataset_path,
    n_train_num,
    n_val_num,
    n_test_num,
    train_ratio=None,
    test_ratio=None,
):
    if config.data_module == "SAE":
        assert config.sdf_spatial_resolution is not None
        return SAEDataModule(
            dataset_path,
            out_keys=config.out_keys,
            out_channels=config.out_channels,
            n_train=n_train_num,
            n_val=n_val_num,
            n_test=n_test_num,
            spatial_resolution=config.sdf_spatial_resolution,
            lazy_loading=config.lazy_loading,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
        )
    else:
        raise NotImplementedError(f"Unknown datamodule: {config.data_module}")


def instantiate_inferencedatamodule(config, dataset_path, bounds_dir, n_inference_num):
    if config.data_module == "SAE":
        assert config.sdf_spatial_resolution is not None
        return SAEInferenceDataModule(
            dataset_path,
            out_keys=config.out_keys,
            out_channels=config.out_channels,
            n_inference=n_inference_num,
            spatial_resolution=config.sdf_spatial_resolution,
            lazy_loading=config.lazy_loading,
            bounds_dir=bounds_dir,
        )
    else:
        raise NotImplementedError(f"Unknown datamodule: {config.data_module}")
