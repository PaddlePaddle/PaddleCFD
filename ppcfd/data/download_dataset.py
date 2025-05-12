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


import paddle
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
import subprocess
import wget
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Literal, NoReturn, Optional, Tuple, Union
import os 
import subprocess
os.environ['WGET_USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

@dataclass
class DataCard():
    HF_OWNER: str
    HF_PREFIX: str
    INDEX: int
    LENGTH: int
    FILE_TYPES: int
    RUN_LOCAL_DIR: Path
    MIRROR: str = field(default="hf-mirror.com", init=True)

    def set_index(self, i):
        self.INDEX = i+1
        self.RUN_LOCAL_DIR = Path(f"run_{self.INDEX}")

def sub_wget(command):
    os.system(filename)


def get_free_space_in_gb(local_dir):
    # get disk space
    usage = shutil.disk_usage(local_dir)
    # to GB
    free_space_gb = usage.free / (1024 ** 3)
    return free_space_gb


def _dfc_hub_download_to_local_dir(
    *,
    # Destination
    local_dir: Union[str, Path],
    # File info
    repo_id: str,
    repo_type: str,
    filename: str,
    revision: str,
    # HTTP info
    endpoint: Optional[str],
    etag_timeout: float,
    headers: Dict[str, str],
    proxies: Optional[Dict],
    token: Union[bool, str, None],
    # Additional options
    cache_dir: str,
    force_download: bool,
    local_files_only: bool,
    estimated_disk_space = 30,
    max_workers: int = 5,
) -> str:
    """Download a given file to a local folder, if not already present."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    def _check_disk_space(estimated_disk_space, local_dir):
        target_dir_free = get_free_space_in_gb(local_dir)
        if estimated_disk_space > target_dir_free:
            raise ValueError(f"Insufficient disk space [{estimated_disk_space:.1f} GB] on your machine for this operation. The estimated required space is [{target_dir_free:.1f} GB]. Please make sure you have enough free disk space to download the file. ")
    _check_disk_space(estimated_disk_space, local_dir)

    data = DataCard(HF_OWNER="neashton", HF_PREFIX="drivaerml", INDEX=0, LENGTH=500, FILE_TYPES=5, RUN_LOCAL_DIR="")
    file_types_list = [[ ] for _ in range(data.FILE_TYPES)]
    for i in range(data.LENGTH):
        data.set_index(i)
        local_dir_subdir = local_dir / data.RUN_LOCAL_DIR
        Path(local_dir_subdir).mkdir(parents=True, exist_ok=True)
        file_types_list[0].append(['wget', '-c', '-O', (local_dir_subdir/ Path(f"force_mom_{data.INDEX}.csv",)).as_posix() , f"https://{data.MIRROR}/datasets/{data.HF_OWNER}/{data.HF_PREFIX}/resolve/main/{data.RUN_LOCAL_DIR.as_posix()}/force_mom_{data.INDEX}.csv"])
        # uncomment for more data
        # file_types_list[1].append(['wget', '-c', '-O', (local_dir_subdir/ Path(f"drivaer_{data.INDEX}.stl",)).as_posix() , f"https://{data.MIRROR}/datasets/{data.HF_OWNER}/{data.HF_PREFIX}/resolve/main/{data.RUN_LOCAL_DIR.as_posix()}/drivaer_{data.INDEX}.stl"])
        # file_types_list[2].append(['wget', '-c', '-O', (local_dir_subdir/ Path(f"boundary_{data.INDEX}.vtp",)).as_posix() , f"https://{data.MIRROR}/datasets/{data.HF_OWNER}/{data.HF_PREFIX}/resolve/main/{data.RUN_LOCAL_DIR.as_posix()}/boundary_{data.INDEX}.vtp"])
        # file_types_list[3].append(['wget', '-c', '-O', (local_dir_subdir/ Path(f"volume_{data.INDEX}.vtu.00.part",)).as_posix() , f"https://{data.MIRROR}/datasets/{data.HF_OWNER}/{data.HF_PREFIX}/resolve/main/{data.RUN_LOCAL_DIR.as_posix()}/volume_{data.INDEX}.vtu.00.part"])
        # file_types_list[4].append(['wget', '-c', '-O', (local_dir_subdir/ Path(f"volume_{data.INDEX}.vtu.01.part",)).as_posix() , f"https://{data.MIRROR}/datasets/{data.HF_OWNER}/{data.HF_PREFIX}/resolve/main/{data.RUN_LOCAL_DIR.as_posix()}/volume_{data.INDEX}.vtu.01.part"])
    
    for command_list in file_types_list:
        # concurrency limit
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(subprocess.run, cmd) for cmd in command_list]
            for future in futures:
                future.result()  # handle exception


def dfc_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = 1000,
    token: Union[bool, str, None] = None,
    local_files_only: bool = False,
    headers: Optional[Dict[str, str]] = None,
    endpoint: Optional[str] = None,
    resume_download: Optional[bool] = None,
    force_filename: Optional[str] = None,
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    max_workers: int = 5,
) -> str:
    def build_headers():
        return None
    return _dfc_hub_download_to_local_dir(
                # Destination
                local_dir=local_dir,
                # File info
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                revision=revision,
                # HTTP info
                endpoint=endpoint,
                etag_timeout=etag_timeout,
                headers=build_headers(),
                proxies=proxies,
                token=token,
                # Additional options
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                max_workers=max_workers
            )
std = time.time()
# Download from PPCFD-hub
dfc_hub_download(
    repo_id="DNNFluid-Car/DrivAerML",  #  repo sign
    filename=None,                     #  specify certain files to download
    repo_type="dataset",               #  specify download type（option：dataset/model...）
    local_dir=Path("./downloaded_dataset"),  # local saving dir（Pathlib）
    max_workers=1                      #  concurrency == 5
)
print("Time taken: ", time.time()-std,"seconds")
# 1 process   for 15 files, max_workers=1 : Time taken:  9.841650009155273 seconds
# 5 processes for 15 files, max_workers=5 : Time taken:  2.601330518722534 seconds
