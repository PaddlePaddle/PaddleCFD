import paddle


# 在 PaddlePaddle 中继承自 paddle.io.Dataset
class BaseDataset(paddle.io.Dataset):
    """
    同样的类文档描述...
    """

    def __init__(
        self,
        data_dir: str,
    ):
        """
        构造函数参数与 PyTorch 类似...
        """
        super(BaseDataset, self).__init__()  # 注意调用父类构造函数的方式
        self.dataset = None
        self._check_args()

    def _check_args(self):
        """检查参数是否有效。"""
        pass

    def __len__(self):
        # 返回数据集大小
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回数据项
        X, y = self.dataset
        return X[idx], y[idx]
