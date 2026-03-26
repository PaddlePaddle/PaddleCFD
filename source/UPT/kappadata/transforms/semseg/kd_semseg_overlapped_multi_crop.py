from kappadata.compat.image import get_image_size, crop

from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform
from kappadata.utils.param_checking import to_2tuple
import paddle

class KDSemsegOverlappedMultiCrop(KDStochasticTransform):
    def __init__(self, crop_size, overlap=0.5, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = to_2tuple(crop_size)
        if overlap != 0.5:
            raise NotImplementedError("overlaps other than 0.5 require different logic in __call__")
        self.overlap = to_2tuple(overlap)
        assert (self.crop_size[0] * self.overlap[0]).is_integer()
        assert (self.crop_size[1] * self.overlap[1]).is_integer()

    def __call__(self, xsemseg, ctx=None):
        x, semseg = xsemseg
        width, height = get_image_size(x)
        crop_height, crop_width = self.crop_size
        assert height % crop_height == 0
        assert width % crop_width == 0

        x_crops = []
        semseg_crops = []
        overlap_height = int(crop_height * self.overlap[0])
        overlap_width = int(crop_width * self.overlap[1])
        num_rows = 1 + (height - crop_height) // overlap_height
        num_cols = 1 + (width - crop_width) // overlap_width
        for i in range(num_rows):
            for j in range(num_cols):
                cur_top = i * overlap_height
                cur_left = j * overlap_width
                x_crops.append(crop(x, top=cur_top, left=cur_left, height=crop_height, width=crop_width))
                semseg_crops.append(crop(semseg, top=cur_top, left=cur_left, height=crop_height, width=crop_width))
        x_crops = paddle.stack(x_crops)
        semseg_crops = paddle.stack(semseg_crops)
        return x_crops, semseg_crops
