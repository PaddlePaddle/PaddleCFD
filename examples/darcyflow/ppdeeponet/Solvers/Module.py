import paddle
from sklearn.model_selection import train_test_split


class MyDataset(paddle.io.Dataset):

    def __init__(self, x: paddle.to_tensor, y: paddle.to_tensor, z: paddle.to_tensor = None):
        """
        Input:
            x: size(N,?)
            y: size(N,?)
            z: size(N,?)
        """
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, index):
        if self.z is not None:
            return self.x[index], self.y[index], self.z[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return tuple(self.x.shape)[0]


class MyIndex(paddle.io.Dataset):

    def __init__(self, index: paddle.to_tensor):
        """
        Input:
            index: size(N,)
        """
        self.index = index

    def __getitem__(self, idx):
        return self.index[idx]

    def __len__(self):
        return tuple(self.index.shape)[0]


class Solver:

    def __init__(self, dtype="float32"):
        """Neural Operator-based PDE solver"""
        self.dtype = dtype

    def datasplit(self, x, y, test_rate_or_size: float = 0.2):
        """Split the data into training set and testing set
        Input:
            x: size(?,d)
            y: size(?,d)
            test_rate_or_size: the rate/size of testing set
        Output:
            x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate_or_size)
        print(f"The train_data size: {tuple(x_train.shape)}, {tuple(y_train.shape)}")
        print(f"The test_data size: {tuple(x_test.shape)}, {tuple(y_test.shape)}")
        return x_train, x_test, y_train, y_test

    def dataloader(
        self, x: paddle.to_tensor, y: paddle.to_tensor, z: paddle.to_tensor = None, batch_size: int = 100, shuffle=True
    ):
        """Prepare the data_loader for training
        Input:
            x: size(N,?)
            y: size(N,?)
            z: size(N,?)
            batch_size: int
        Output:
            train_loader
        """
        return paddle.io.DataLoader(dataset=MyDataset(x, y, z), batch_size=batch_size, shuffle=shuffle)

    def indexloader(self, N: int, batch_size: int = 100, shuffle=True):
        """Prepare the index_loader for training
        Input:
            index: int
        """
        index = paddle.to_tensor(data=[i for i in range(N)], dtype="int32")
        return paddle.io.DataLoader(dataset=MyIndex(index), batch_size=batch_size, shuffle=shuffle)


class LossClass(object):

    def __init__(self, solver: Solver, **kwrds):
        self.solver = solver

    def Loss_beta(self):
        """The loss of beta model"""
        return paddle.to_tensor(data=0.0, dtype=self.solver.dtype, place=self.solver.place)

    def Loss_pde(self):
        """The loss of pde"""
        return paddle.to_tensor(data=0.0, dtype=self.solver.dtype, place=self.solver.place)

    def Loss_data(self):
        """The loss of boundary conditions"""
        return paddle.to_tensor(data=0.0, dtype=self.solver.dtype, place=self.solver.place)

    def Error(self):
        """The errors"""
        return paddle.to_tensor(data=0.0, dtype=self.solver.dtype, place=self.solver.place)
