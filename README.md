# PaddleCFD

## About PaddleCFD

PaddleCFD is a deep learning toolkit for surrogate modeling, equation discovery, shape optimization and flow-control strategy discovery in the field of fluid mechanics. Currently, it mainly supports surrogate modeling, including models based on fourier neural operator (FNO), diffusion model, transformer, physics-informed model and Kolmogorov-Arnold Networks (KAN).

## Code structure

- `config`: config files for different tasks
- `doc`: documentation
- `examples`: example scripts
- `ppcfd/data`: data-process source code
- `ppcfd/model`: model source code
- `ppcfd/utils`: util code

## How to run

### Installation

```bash
pip install paddlepaddle
pip install paddlecfd
```

## ppcfd/data

### basic functions

| file source  | usage | applicable scenarios | functions | support |
|:--| :--| :--| :--| :--|
| public dataset | load_dataset("https://paddle-org.bj.bcebos.com/paddlecfd/datasets/ppdiffusion/ns_trajectory_dataset/single/ns-runs_eval-cors1-navier-stokes-n5-t65-n0_tagcors1_00001.h5") | officially hosted standard datasets | automatic download‌ | ✅ |
| local dataset | load_dataset("./burgers.mat") | your own datasets | multiple formats are supported | ✅ |
| datasets with mixed formats | directory = "./mixed_data/"; full_paths = [os.path.join(directory, entry) for entry in os.listdir(directory)]; load_dataset(full_paths) | multi-format datasets storage | automatically identify formats | ✅ |

### architecture

```
ppcfd/data/
├── __init__.py
├── downloader.py    # data download module
├── loader.py        # data loading module
└── parser
    ├── __init__.py
    ├── base_parser.py    # basic data analysis module
    ├── h5_mat2npz.py    # implementation: .h5/.mat to .npz
    └── ......    # implementation: xxx to .npz
```

### in action

```python
from ppcfd.data import load_dataset
from ppcfd.data.parser import MatTransition


if __name__ == "__main__":
    path = "./burgers.mat"

    print("———— test: loading dataset with `load_dataset`————")
    dataset = load_dataset(path=path)

    # using url as path is not supported in this way
    print("———— test: loading dataset directly with corresponding Class————")
    trans_obj = MatTransition(path)
```
