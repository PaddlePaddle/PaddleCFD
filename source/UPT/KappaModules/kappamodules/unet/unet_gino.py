import paddle


class UnetGino(paddle.nn.Layer):
    def __init__(self, input_dim, hidden_dim, depth=4, num_groups=8, output_dim=None):
        super().__init__()
        self.dim = hidden_dim
        self.depth = depth
        self.num_groups = num_groups
        self.output_dim = output_dim
        assert depth > 1
        assert isinstance(hidden_dim, int) and hidden_dim % 2 == 0
        dim_per_level = [(hidden_dim * 2**i) for i in range(depth)]
        self.down_blocks = paddle.nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.down_blocks.append(
                    paddle.nn.Sequential(
                        paddle.nn.Identity(),
                        paddle.nn.GroupNorm(num_groups=1, num_channels=input_dim),
                        paddle.nn.Conv3d(
                            input_dim,
                            dim_per_level[0] // 2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        paddle.nn.ReLU(),
                        paddle.nn.GroupNorm(
                            num_groups=num_groups, num_channels=dim_per_level[0] // 2
                        ),
                        paddle.nn.Conv3d(
                            dim_per_level[0] // 2,
                            dim_per_level[0],
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        paddle.nn.ReLU(),
                    )
                )
            else:
                self.down_blocks.append(
                    paddle.nn.Sequential(
                        paddle.nn.MaxPool3D(kernel_size=2, stride=2),
                        paddle.nn.GroupNorm(
                            num_groups=num_groups, num_channels=dim_per_level[i] // 2
                        ),
                        paddle.nn.Conv3d(
                            dim_per_level[i] // 2,
                            dim_per_level[i] // 2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        paddle.nn.ReLU(),
                        paddle.nn.GroupNorm(
                            num_groups=num_groups, num_channels=dim_per_level[i] // 2
                        ),
                        paddle.nn.Conv3d(
                            dim_per_level[i] // 2,
                            dim_per_level[i],
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        paddle.nn.ReLU(),
                    )
                )
        self.up_blocks = paddle.nn.ModuleList()
        rev_dim_per_level = list(reversed(dim_per_level))
        for i in range(depth - 1):
            self.up_blocks.append(
                paddle.nn.Sequential(
                    paddle.nn.GroupNorm(
                        num_groups=num_groups,
                        num_channels=rev_dim_per_level[i] + rev_dim_per_level[i + 1],
                    ),
                    paddle.nn.Conv3d(
                        rev_dim_per_level[i] + rev_dim_per_level[i + 1],
                        rev_dim_per_level[i + 1],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    paddle.nn.ReLU(),
                    paddle.nn.GroupNorm(
                        num_groups=num_groups, num_channels=rev_dim_per_level[i + 1]
                    ),
                    paddle.nn.Conv3d(
                        rev_dim_per_level[i + 1],
                        rev_dim_per_level[i + 1],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    paddle.nn.ReLU(),
                )
            )
        self.pred = paddle.nn.Conv3d(
            rev_dim_per_level[-1], output_dim or hidden_dim, kernel_size=1
        )

    def forward(self, x):
        stack = []
        for down_block in self.down_blocks:
            x = down_block(x)
            stack.append(x)
        stack.pop()
        for up_block in self.up_blocks:
            residual = stack.pop()
            x = paddle.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x = paddle.concat([residual, x], dim=1)
            x = up_block(x)
        x = self.pred(x)
        return x
