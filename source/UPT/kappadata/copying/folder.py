import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from kappadata.utils.logging import log
from .copying_utils import folder_contains_mostly_zips, run_unzip_jobs


@dataclass
class CopyFolderResult:
    was_copied: bool
    was_deleted: bool
    source_format: str


def _check_src_path(src_path):
    if src_path.exists() and src_path.is_dir():
        return True
    if src_path.with_suffix(".zip").exists():
        return True
    return False


def copy_folder_from_global_to_local(
        global_path,
        local_path,
        relative_path=None,
        num_workers=0,
        log_fn=None,
) -> CopyFolderResult:
    if not isinstance(global_path, Path):
        global_path = Path(global_path).expanduser()
    if not isinstance(local_path, Path):
        local_path = Path(local_path).expanduser()

    # check src_path exists (src_path can be folder or .zip)
    src_path = global_path / relative_path if relative_path is not None else global_path
    assert _check_src_path(src_path), f"invalid src_path (can be folder or folder of zips or zip) '{src_path}'"

    # if dst_path exists:
    # - autocopy start/end file exists -> already copied -> do nothing
    # - autocopy start file exists && autocopy end file doesn't exist -> incomplete copy -> delete and copy again
    # - autocopy start file doesn't exists -> manually copied dataset -> do nothing
    dst_path = local_path / relative_path if relative_path is not None else local_path
    start_copy_file = dst_path / "autocopy_start.txt"
    end_copy_file = dst_path / "autocopy_end.txt"
    was_deleted = False
    if dst_path.exists():
        if start_copy_file.exists():
            if end_copy_file.exists():
                # already automatically copied -> do nothing
                log(log_fn, f"dataset was already automatically copied '{dst_path}'")
                return CopyFolderResult(
                    was_copied=False,
                    was_deleted=False,
                    source_format=None,
                )
            else:
                # incomplete copy -> delete and copy again
                log(log_fn, f"found incomplete automatic copy in '{dst_path}' -> deleting folder")
                shutil.rmtree(dst_path)
                was_deleted = True
                dst_path.mkdir()
        else:
            log(log_fn, f"using manually copied dataset '{dst_path}'")
            return CopyFolderResult(was_copied=False, was_deleted=False, source_format=None)
    else:
        dst_path.mkdir(parents=True)

    # create start_copy_file
    with open(start_copy_file, "w") as f:
        f.write("this file indicates that an attempt to copy the dataset automatically was started")

    # copy
    if src_path.exists() and src_path.is_dir():
        contains_mostly_zips, zips = folder_contains_mostly_zips(src_path)
        if contains_mostly_zips:
            source_format = "zips"
            # extract all zip folders into dst (e.g. audioset/train/batch_0.zip)
            log(log_fn, f"extracting {len(zips)} zips from '{src_path}' to '{dst_path}' using {num_workers} workers")
            unzip_batched_zips(src=src_path, dst=dst_path, num_workers=num_workers)
        else:
            source_format = "raw"
            # copy folders which contain the raw files (not zipped or anything)
            log(log_fn, f"copying files of '{src_path}' to '{dst_path}'")
            # copy folder (dirs_exist_ok=True because dst_path is created for start_copy_file)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    elif src_path.with_suffix(".zip").exists():
        source_format = "zip"
        # extract zip
        log(log_fn, f"extracting '{src_path.with_suffix('.zip')}' to '{dst_path}'")
        with zipfile.ZipFile(src_path.with_suffix(".zip")) as f:
            f.extractall(dst_path)
    else:
        raise NotImplementedError

    # create end_copy_file
    with open(end_copy_file, "w") as f:
        f.write("this file indicates that copying the dataset automatically was successful")

    log(log_fn, "finished copying data from global to local")
    return CopyFolderResult(
        was_copied=True,
        was_deleted=was_deleted,
        source_format=source_format,
    )




def unzip_batched_zips(src, dst, num_workers=0):
    src_path = Path(src).expanduser()
    assert src_path.exists(), f"src_path '{src_path}' doesn't exist"
    dst_path = Path(dst).expanduser()
    dst_path.mkdir(exist_ok=True, parents=True)

    # compose jobs
    jobargs = []
    for item in os.listdir(src_path):
        if not item.endswith(".zip"):
            continue
        src_uri = src_path / item
        jobargs.append((src_uri, dst_path))

    # run jobs
    run_unzip_jobs(jobargs=jobargs, num_workers=num_workers)
