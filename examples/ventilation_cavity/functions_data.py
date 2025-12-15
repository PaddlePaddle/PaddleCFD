from typing import List
import numpy as np
import paddle
import pandas as pd

FILTER_CONST = 2
class Dataset(paddle.io.Dataset):
    def __init__( # 相当于变量的赋值，没有外界干扰则保持默认值
        self,
        data_path: str,         # 数据路径
        t_len_in: int = 8,      # 输入序列长度
        t_len_out: int = 1,     # 输出序列长度
        ratio: List = [0.8, 0.2, 0.0],   # 数据集的划分比例
        suffle: bool = True,    # 是否打乱数据
        mode="train",           # 使用模型是训练、验证、测试
        num_conditions: int = 1,  # 工况数量（用于用于处理多工况滑动窗口构建样本的情况）
    ):
        super().__init__()
        self.data_path = data_path
        self.t_len_in = t_len_in
        self.t_len_out = t_len_out
        self.ratio = ratio
        self.mode = mode
        self.num_conditions = num_conditions
        self.load_data()
        self.get_idx(suffle = suffle)
        self.dataset_dict = {"train":self.raw_dataset[self.idx_train],
                             'val':self.raw_dataset[self.idx_val],
                             "test":self.raw_dataset[self.idx_test]}
        # 初始化标准化器
        self.input_scalers = None
        self.label_scalers = None
        # 训练模式拟合标准化器，测试模式加载
        if self.mode == "train":
            self._fit_scalers()
        else:
            self._load_scalers()


    ## 滑动窗口法构造样本
    def load_data(self):
        # 读取完整数据
        full_data = pd.read_csv(self.data_path, header=None, encoding="GBK").to_numpy()

        # 计算每个工况的数据长度
        total_length = full_data.shape[0]
        condition_length = total_length // self.num_conditions

        # 为每个工况创建样本  （这里的两个1表示每隔多少个时间步构造一个样本）
        all_samples = []
        for cond in range(self.num_conditions):
            start_idx = cond * condition_length
            end_idx = (cond + 1) * condition_length
            condition_data = full_data[start_idx:end_idx]

            # 为当前工况创建样本
            num_samples = condition_data.shape[0] - self.t_len_in - self.t_len_out + 1
            condition_samples = np.empty(
                (num_samples, self.t_len_in + self.t_len_out, condition_data.shape[1]),
                dtype=condition_data.dtype
            )

            for i in range(num_samples):
                condition_samples[i] = condition_data[i:i + self.t_len_in + self.t_len_out]

            all_samples.append(condition_samples)
        # 合并所有工况的样本
        self.raw_dataset = np.concatenate(all_samples, axis=0)
        self.num_total = self.raw_dataset.shape[0]
        print(self.num_total, self.raw_dataset.shape)


    # 划分训练集和测试集
    def get_idx(self, suffle=False):
        idx = np.arange(self.num_total)
        num_train, num_val, num_test = (
            int(self.num_total * self.ratio[0]),
            int(self.num_total * self.ratio[1]),
            int(self.num_total * self.ratio[2]),
        )
        self.idx_train, self.idx_val, self.idx_test = (
            idx[:num_train],
            idx[num_train : num_train + num_val],
            idx[num_train + num_val : num_train + num_val + num_test],
        )


    # 计算训练集各变量的均值及方差
    def _fit_scalers(self):
        inputs_all = self.dataset_dict["train"][:, :, :8]
        labels_all = self.dataset_dict["train"][:, :, 8:]
        self.input_means = np.mean(inputs_all, axis=(0,1))
        self.input_stds = np.std(inputs_all, axis=(0,1))
        self.label_mean = np.mean(labels_all)
        self.label_std = np.std(labels_all)
        np.savez(
            "scaler_params.npz",
            input_means=self.input_means,
            input_stds=self.input_stds,
            label_mean=self.label_mean,
            label_std=self.label_std,
        )


    # 加载训练集各变量的均值及方差，以便于后续归一化
    def _load_scalers(self):
        scaler_params = np.load("scaler_params.npz")
        self.input_means = scaler_params["input_means"]
        self.input_stds = scaler_params["input_stds"]
        self.label_mean = scaler_params["label_mean"]
        self.label_std = scaler_params["label_std"]


    def __len__(self):
        return len(getattr(self, f'idx_{self.mode}', 0))

    def __getitem__(self, index):
        inputs = self.dataset_dict[self.mode][index, :self.t_len_in, :8]                                  # 第index个样本的输入序列
        labels = self.dataset_dict[self.mode][index, self.t_len_in: self.t_len_in + self.t_len_out, 8:]   # 第index个样本的输出序列
        # # 归一化
        # inputs = (inputs - self.input_means) / (self.input_stds + 1e-8)
        # labels = (labels - self.label_mean) / (self.label_std + 1e-8)
        # 检查样本输入、输出的形状（必须是三维数组）
        if inputs.ndim == 1:
            inputs = inputs[..., np.newaxis]
        if labels.ndim == 1:
            labels = labels[..., np.newaxis]

        return (
            paddle.to_tensor(inputs, dtype=paddle.get_default_dtype()),
            paddle.to_tensor(labels, dtype=paddle.get_default_dtype()), # 转换数据类型从NumPy数组变为PaddlePaddle Tensor
        )


class DataLoader(paddle.io.DataLoader): # 根据定义的批大小，处理训练、验证、测试样本集，得到用于模型输入、输出的形式（batchsize，时间步数，维数）
    def __init__(self, dataset, enable_ddp=False):  # enable_ddp用于控制数据加载模式
        self.enable_ddp = enable_ddp
        self.dataset = dataset
        # print(self.dataset.shape)

    def dataloader(self, **kwargs):
        if self.enable_ddp is True: # 分布式批次划分
            sampler = paddle.io.DistributedBatchSampler(
                self.dataset,
                rank=paddle.distributed.get_rank(),
                batch_size=kwargs.get("batch_size", 1),
            )
            kwargs.pop("batch_size", None)
            return paddle.io.DataLoader(
                self.dataset,
                batch_sampler=sampler,
                **kwargs,
            )
        else:                       # 非分布式批次划分
            return paddle.io.DataLoader(
                self.dataset,
                **kwargs,
            )
