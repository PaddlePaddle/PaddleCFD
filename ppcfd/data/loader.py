# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ppcfd.data.downloader import download
from ppcfd.data.parser import DataParserFactory
from ppcfd.utils.logger import logger


def load_dataset(
    path: Union[str, List[str]],
    input_format: str = "auto",
    cache_dir: Optional[Union[str, Path]] = None,
    force_redownload: bool = False,
    use_parallel: bool = False,
    max_workers: int = 4,
    **parser_kwargs,
) -> List:
    if not isinstance(path, list):
        path = [path]

    if cache_dir and not isinstance(cache_dir, Path):
        cache_dir = Path(cache_dir)

    # Automatic identification of remote resources
    if path[0].startswith(("http://", "https://")):
        local_paths = download(path, cache_dir, force_redownload, use_parallel, max_workers)
    else:
        local_paths = [Path(p) for p in path]

    if input_format == "auto":
        formats = [p.suffix[1:].lower() for p in local_paths]
    else:
        formats = [input_format.lower()] * len(local_paths)

    datasets = []
    for local_path, fmt in zip(local_paths, formats):
        if fmt in ["npz", "npy"]:
            logger.info(f"The data format of {local_path} is detected to be npz or npy.")
            datasets.append(np.load(local_path))
        parser_kwargs["file_path"] = local_path
        loader = DataParserFactory.get_loader(fmt, **parser_kwargs)
        datasets.append(loader.get_data())
    return datasets
