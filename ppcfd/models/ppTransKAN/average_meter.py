# 这部分代码主要用于训练过程中更新loss，accuracy等指标参数的取值
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):  # 将所有变量置零
        self.val = 0 # 当前值
        self.avg = 0 # 平均值
        self.sum = 0 # 总和
        self.count = 0 # 总观测？？？次数

    def update(self, val, n=1): # 更新变量值
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Average Meter with dictionary values
class AverageMeterDict:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val, n=1): # 这里的n即表示每个batch_size的loss用于计算平均值时的权重，n=1表示权重一致
        for k, v in val.items():
            if k not in self.val:
                self.val[k] = 0
                self.sum[k] = 0
                self.count[k] = 0
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]
