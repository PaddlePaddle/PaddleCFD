import numpy as np
import os
import shutil
import zipfile
from pathlib import Path


def create_zips_imagefolder(src, dst):
    """
    creates a zip for each class
    Source:
    imagenet1k/train/n2933412
    imagenet1k/train/n3498534
    Result:
    imagenet1k/train/n2933412.zip
    imagenet1k/train/n3498534.zip
    """
    src_path = Path(src).expanduser()
    assert src_path.exists(), f"src_path '{src_path}' doesn't exist"
    dst_path = Path(dst).expanduser()
    dst_path.mkdir(exist_ok=True, parents=True)

    for item in os.listdir(src_path):
        src_uri = src_path / item
        if not src_uri.is_dir():
            continue
        shutil.make_archive(
            base_name=dst_path / item,
            format="zip",
            root_dir=src_uri,
        )


def create_zips_folder(src, dst, batch_size=1000):
    """
    creates zips where each zip has <batch_size> samples
    Source:
    audioset/train/sample0.wav
    audioset/train/sample1.wav
    audioset/train/sample2.wav
    Result:
    audioset/train/batch_0.zip
    audioset/train/batch_1.zip
    """
    src_path = Path(src).expanduser()
    assert src_path.exists(), f"src_path '{src_path}' doesn't exist"
    dst_path = Path(dst).expanduser()
    dst_path.mkdir(exist_ok=True, parents=True)

    # retrieve items and check validity
    items = []
    for item in os.listdir(src_path):
        assert (src_path / item).is_file(), f"source folder has to contain only files ({item})"
        assert not item.endswith(".zip"), f"source folder cant contain zips ({item})"
        items.append(item)

    # create zipped folders, each with <batch_size> items
    num_batches = (len(items) + batch_size - 1) // batch_size
    num_digits = int(np.log10(num_batches)) + 1
    format_str = f"{{:0{num_digits}d}}"
    for i in range(0, num_batches):
        with zipfile.ZipFile(dst / f"batch_{format_str.format(i)}.zip", "w") as f:
            for j in range(i * batch_size, min(len(items), (i + 1) * batch_size)):
                f.write(src_path / items[j], items[j])
