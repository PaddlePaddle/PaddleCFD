import paddle


class DoubleConv(paddle.nn.Layer):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=mid_channels),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=out_channels),
            paddle.nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpSampler(paddle.nn.Layer):
    """Up-scaling then double conv"""

    def __init__(self, in_channels, out_channels, mode="conv", scale_factor=2):
        super().__init__()
        h_channels = (in_channels + out_channels) // 2
        if mode in ["conv", "convolution"]:
            self.up = paddle.nn.Conv2DTranspose(
                in_channels=in_channels,
                out_channels=h_channels,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(h_channels, out_channels)
        else:
            # if {bilinear, nearest,..}, use the normal convolutions to reduce the number of channels
            align_corners = None if mode == "nearest" else True  # align_corners does not work for nearest neighbor
            self.up = paddle.nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
            self.conv = DoubleConv(in_channels, out_channels, h_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)  # (B, C_in, H_out, W_out)
        x1 = self.conv(x1)  # (B, C_out, H_out, W_out)
        return x1
