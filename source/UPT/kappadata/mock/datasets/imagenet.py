import paddle
import yaml
from kappadata.compat.image import to_pil_image

from kappadata.utils.param_checking import to_2tuple
from .utils import setup


def generate_mock_imagenet(
        dst,
        samples_per_class=13,
        resolution_min=8,
        resolution_max=32,
        seed=0,
        log_fn=None,
):
    log, dst, generator = setup(log_fn=log_fn, dst=dst, seed=seed)
    height_min, width_min = to_2tuple(resolution_min)
    height_max, width_max = to_2tuple(resolution_max)

    with open("kappadata/res/imagenet_class_ids.yaml") as f:
        folder_names = yaml.safe_load(f)

    log(f"generating mock ImageNet into '{dst.as_posix()}'")
    for split in ["train", "val"]:
        split_uri = dst / split
        split_uri.mkdir()
        log(f"generating {split} samples ({samples_per_class} samples_per_class)")
        for i, folder_name in enumerate(folder_names):
            folder_uri = split_uri / folder_name
            folder_uri.mkdir()
            for j in range(samples_per_class):
                if split == "train":
                    fname = f"{folder_name}_{j}.JPEG"
                elif split == "val":
                    fname = f"ILSVRC2012_val_{i * samples_per_class + j}.JPEG"
                else:
                    raise NotImplementedError

                height = paddle.randint(height_min, height_max, size=(1,), generator=generator)
                width = paddle.randint(width_min, width_max, size=(1,), generator=generator)
                img = paddle.rand((3, height, width))
                to_pil_image(img).save(folder_uri / fname)
