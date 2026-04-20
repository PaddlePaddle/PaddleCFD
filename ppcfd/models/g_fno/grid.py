import paddle


class grid(paddle.nn.Module):
    def __init__(self, twoD, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric", "None"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.include_grid = grid_type != "None"
        self.grid_dim = (1 + (not self.symmetric) + (not twoD)) * self.include_grid
        if self.include_grid:
            if twoD:
                self.get_grid = self.twoD_grid
            else:
                self.get_grid = self.threeD_grid
        else:
            self.get_grid = paddle.nn.Identity()

    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = (
            paddle.linspace(0, 1, size_x)
            .reshape(1, size_x, 1, 1)
            .repeat([batchsize, 1, size_y, 1])
        )
        gridy = (
            paddle.linspace(0, 1, size_y)
            .reshape(1, 1, size_y, 1)
            .repeat([batchsize, size_x, 1, 1])
        )
        if not self.symmetric:
            grid = paddle.cat((gridx, gridy), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid.to(x.device)
        return paddle.cat((x, grid), dim=-1)

    def threeD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = (
            paddle.linspace(0, 1, size_x)
            .reshape(1, size_x, 1, 1, 1)
            .repeat([batchsize, 1, size_y, size_z, 1])
        )
        gridy = (
            paddle.linspace(0, 1, size_y)
            .reshape(1, 1, size_y, 1, 1)
            .repeat([batchsize, size_x, 1, size_z, 1])
        )
        gridz = (
            paddle.linspace(0, 1, size_z)
            .reshape(1, 1, 1, size_z, 1)
            .repeat([batchsize, size_x, size_y, 1, 1])
        )
        if not self.symmetric:
            grid = paddle.cat((gridx, gridy, gridz), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = paddle.cat((gridx + gridy, gridz), dim=-1)
        grid = grid.to(x.device)
        return paddle.cat((x, grid), dim=-1)
