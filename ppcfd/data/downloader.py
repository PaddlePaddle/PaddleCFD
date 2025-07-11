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


import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import requests

from ppcfd.utils.logger import logger


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".tmp"


def download(
    urls: List[str],
    cache_dir: Optional[Path] = None,
    force_redownload: bool = False,
    use_parallel: bool = False,
    max_workers: int = 4,
) -> List[Path]:
    if cache_dir is None:
        logger.info(f"`cache_dir` is not set, using default dir path {DEFAULT_CACHE_DIR}")
    if use_parallel:
        cpu_count = os.cpu_count()
        if max_workers > cpu_count:
            recommended_workers = max(0, int(cpu_count * 0.5) - 1)
            logger.warning(
                f"The max workers is set to {max_workers} which has exceeded the number of CPU cores {cpu_count}, "
                f"the max_workers is automatically set to the recommended {recommended_workers}(50% of CPU cores)"
            )
        return download_parallel(urls, cache_dir, force_redownload, max_workers)
    else:
        paths = []
        for url in urls:
            paths.append(download_from_url(url, cache_dir, force_redownload))
        return paths


def download_from_url(
    url: str,
    cache_dir: Optional[Path] = None,
    force_redownload: bool = False,
) -> Path:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    filepath = cache_dir / filename

    if filepath.exists() and not force_redownload:
        return filepath

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        raise RuntimeError(f"Download failed: {str(e)}")

    return filepath


def download_parallel(
    urls: List[str],
    cache_dir: Optional[Path] = None,
    force_redownload: bool = False,
    max_workers: int = 4,
) -> List[Path]:
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(download_from_url, url, cache_dir, force_redownload) for url in urls]
        return [f.result() for f in futures]
