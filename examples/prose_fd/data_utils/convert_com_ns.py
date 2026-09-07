import os

import h5py
import numpy as np


def convert(folder, save_folder):
    files = os.listdir(folder)
    print(files)
    print()
    for file in files:
        if "512" not in file:
            continue
        target_file = file.replace("512", "128")
        path = os.path.join(folder, file)
        target_path = os.path.join(save_folder, target_file)
        with h5py.File(path, "r") as f:
            with h5py.File(target_path, "w") as g:
                for k in f.keys():
                    if k == "t-coordinate":
                        g.create_dataset(k, data=np.array(f[k]))
                    elif k.endswith("coordinate"):
                        v = np.mean(np.array(f[k]).reshape(-1, 4), axis=1)
                        g.create_dataset(k, data=v)
                    else:
                        data = np.array(f[k])
                        data = data.reshape(
                            data.shape[0],
                            data.shape[1],
                            data.shape[2] // 4,
                            4,
                            data.shape[3] // 4,
                            4,
                        ).mean(axis=(-3, -1))
                        g.create_dataset(k, data=data)
        print(f"Converted source ({path}) into target ({target_path})\n\n")


if __name__ == "__main__":
    folder = "/data/shared/dataset/pdebench/2D/CFD/2D_Train_Rand/raw_512"
    save_folder = "/data/shared/dataset/pdebench/2D/CFD/2D_Train_Rand"
    convert(folder, save_folder)
    folder = "/data/shared/dataset/pdebench/2D/CFD/2D_Train_Turb/raw_512"
    save_folder = "/data/shared/dataset/pdebench/2D/CFD/2D_Train_Turb"
    convert(folder, save_folder)
