import os
import zipfile

import joblib


def folder_contains_mostly_zips(path):
    # check if subfolders are zips (allow files such as a README inside the folder)
    items = os.listdir(path)
    zips = [item for item in items if item.endswith(".zip")]
    contains_mostly_zips = len(zips) > 0 and len(zips) >= len(items) // 2
    return contains_mostly_zips, zips


def unzip(src, dst):
    with zipfile.ZipFile(src) as f:
        f.extractall(dst)


def run_unzip_jobs(jobargs, num_workers):
    if num_workers <= 1:
        for src, dst in jobargs:
            unzip(src, dst)
    else:
        jobs = [joblib.delayed(unzip)(src, dst) for src, dst in jobargs]
        pool = joblib.Parallel(n_jobs=num_workers)
        pool(jobs)
