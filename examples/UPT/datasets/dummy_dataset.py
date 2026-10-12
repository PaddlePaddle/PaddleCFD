import paddle
from .base.dataset_base import DatasetBase
import numpy as np
from PIL import Image
import paddle

def paddle_to_pil(tensor):
    # 1. 将 Tensor 移动到 CPU 并转为 NumPy
    # 假设输入是 [C, H, W]，且在 0-1 之间
    img_array = tensor.numpy()
    
    # 2. 如果是 [C, H, W] 格式，需转为 PIL 需要的 [H, W, C]
    if len(img_array.shape) == 3:
        img_array = img_array.transpose((1, 2, 0))
    
    # 3. 如果数据是 0.0 - 1.0 之间，需转回 0-255 整数
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
        
    return Image.fromarray(img_array)

class DummyDataset(DatasetBase):
    def __init__(
        self,
        x_shape,
        size=None,
        n_classes=10,
        n_abspos=10,
        is_multilabel=False,
        to_image=False,
        semi_percent=None,
        num_timesteps=10,
        force_timestep_zero=False,
        mode="on-the-fly",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.size = size
        self.x_shape = x_shape
        self._n_classes = n_classes
        self.n_abspos = n_abspos
        self._is_multilabel = is_multilabel
        self.to_image = to_image
        self.semi_percent = semi_percent
        self.num_timesteps = num_timesteps
        self.force_timestep_zero = force_timestep_zero
        self.mode = mode
        assert semi_percent is None or 0.0 <= semi_percent <= 1.0
        assert mode in ["on-the-fly", "preloaded"]
        if self.mode == "preloaded":
            self.x = paddle.randn(len(self), *self.x_shape)
            self.y = paddle.randint(
                low=0, high=max(2, self.getdim_class()), shape=(1,)
            ).tolist()
        else:
            self.x = None
            self.y = None

    def __len__(self):
        return self.size or 131072

    def getitem_x(self, idx, ctx=None):
        if self.x is not None:
            return self.x[idx]
        x = paddle.randn(*self.x_shape)
        if self.to_image:
            x = paddle_to_pil(x)
        return x

    def getitem_timestep(self, idx, ctx=None):
        if self.force_timestep_zero:
            return 0
        max_timestep = self.num_timesteps - self.x_shape[0]
        timestep = paddle.randint(low=0, high=max_timestep, shape=(1,))
        return timestep

    def getshape_timestep(self):
        return (self.num_timesteps,)

    def getshape_class(self):
        return (self._n_classes,) if self._n_classes > 2 else (1,)

    def getitem_class(self, idx, ctx=None):
        if self.semi_percent is not None and idx / len(self) < self.semi_percent:
            return -1
        return self.getitem_class_all(idx, ctx=ctx)

    def getitem_class_all(self, idx, ctx=None):
        if self.y is not None:
            return self.y[idx]
        return paddle.randint(
            low=0, high=max(2, self.getdim_class()), shape=(1,)
        ).item()

    def getall_class(self):
        return [self.getitem_class(i) for i in range(len(self))]

    def getshape_abspos(self):
        return (self.n_abspos,)

    def getitem_abspos(self, idx, ctx=None):
        return paddle.randint(low=0, high=self.n_abspos, shape=(1,)).item()

    def getitem_semseg(self, idx, ctx=None):
        assert len(self.x_shape) == 3
        return paddle.randint(low=0, high=10, shape=self.x_shape[1:])

    @staticmethod
    def getshape_semseg():
        return (10,)

    @property
    def is_multilabel(self):
        return self._is_multilabel
