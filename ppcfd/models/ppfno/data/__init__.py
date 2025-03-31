from .datamodule_lazy import SAEDataModule


def instantiate_datamodule(config, n_train_num, n_val_num, n_test_num):
    if config.data_module == 'SAE':
        assert config.sdf_spatial_resolution is not None
        return SAEDataModule(config.data_path, out_keys=config.out_keys,
            out_channels=config.out_channels, n_train=n_train_num, n_val=
            n_val_num, n_test=n_test_num, spatial_resolution=config.
            sdf_spatial_resolution, lazy_loading=config.lazy_loading,
            only_save_sdf=config.only_save_sdf)
    else:
        raise NotImplementedError(f'Unknown datamodule: {config.data_module}')
