import numpy as np
import paddle
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from KappaModules.kappamodules.convolution import ConvNext
from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import \
    ExcludeFromWdByNameModifier


class RansGridConvnext(SingleModelBase):
    def __init__(
        self,
        patch_size,
        dims,
        depths,
        kernel_size=7,
        depthwise=True,
        global_response_norm=True,
        drop_path_rate=0.0,
        drop_path_decay=False,
        add_pos_tokens=False,
        upsample_size=None,
        upsample_mode="nearest",
        resolution=None,
        concat_pos_to_sdf=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.dims = dims
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.add_pos_tokens = add_pos_tokens
        self.upsample_size = upsample_size
        self.upsample_mode = upsample_mode
        self.resolution = (
            resolution or self.data_container.get_dataset().grid_resolution
        )
        self.ndim = len(self.resolution)
        concat_pos_to_sdf = (
            concat_pos_to_sdf or self.data_container.get_dataset().concat_pos_to_sdf
        )
        if concat_pos_to_sdf:
            input_dim = 4
        else:
            input_dim = 1
        self.model = ConvNext(
            patch_size=patch_size,
            input_dim=input_dim,
            dims=dims,
            depths=depths,
            ndim=self.ndim,
            drop_path_rate=drop_path_rate,
            drop_path_decay=drop_path_decay,
            kernel_size=kernel_size,
            depthwise=depthwise,
            global_response_norm=global_response_norm,
        )
        out_resolution = [
            (r // 2 ** (len(depths) - 1) // patch_size) for r in self.resolution
        ]
        num_output_tokens = int(np.prod(out_resolution))
        if add_pos_tokens:
            # self.pos_tokens = paddle.nn.Parameter(
            #     paddle.empty(size=(1, num_output_tokens, dims[-1]))
            # )
            self.pos_tokens = self.create_parameter(
            shape=[1, num_output_tokens, dims[-1]],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=0.02)
        )
        else:
            self.pos_tokens = None
        # self.type_token = paddle.nn.Parameter(paddle.empty(size=(1, 1, dims[-1])))
        self.type_token = self.create_parameter(
            shape=[1, 1, dims[-1]],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal(std=0.02) # 推荐给由 token 类型的参数一个正态分布初始化
        )
        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = self.ndim
        self.output_shape = num_output_tokens, dims[-1]

    def model_specific_initialization(self):
        if self.add_pos_tokens:
            # paddle.nn.init.trunc_normal_(self.pos_tokens)
            paddle.nn.initializer.TruncatedNormal(std=0.02)(self.pos_tokens)
        # paddle.nn.init.trunc_normal_(self.type_token)
        paddle.nn.initializer.TruncatedNormal(std=0.02)(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        modifiers = [ExcludeFromWdByNameModifier(name="type_token")]
        if self.add_pos_tokens:
            modifiers += [ExcludeFromWdByNameModifier(name="pos_tokens")]
        return modifiers

    def forward(self, x):
        x = paddle.transpose(x, perm=[0, 4, 1, 2, 3])
        if self.upsample_size is not None:
            target_size = self.upsample_size
            if isinstance(target_size, int):
                target_size = [target_size] * self.ndim

            if self.upsample_mode == "nearest":
                x = paddle.nn.functional.interpolate(
                    x, size=target_size, mode=self.upsample_mode
                )
            else:
                x = paddle.nn.functional.interpolate(
                    x,
                    size=target_size,
                    mode=self.upsample_mode,
                    align_corners=True,
                )
        x = self.model(x)
        x = paddle.transpose(x, perm=[0, 2, 3, 4, 1])
        x = paddle.flatten(x, start_axis=1, stop_axis=3)
        x = x + self.type_token
        if self.add_pos_tokens:
            x = x + self.pos_tokens.expand(len(x), -1, -1)
        return x
