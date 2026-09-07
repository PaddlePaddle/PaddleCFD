# Prepare Data

## Step 1. Download the data. 

The dataset we used are collected from [PDEBench](https://github.com/pdebench/PDEBench), [PDEArena](https://github.com/pdearena/pdearena), and [CFDBench](https://github.com/luo-yining/CFDBench). The downloading instructions can be found in the corresponding repositories. 

## Step 2. Preprocess the data.
For this repository, we require all datasets to have the same space dimensions (128x128). However, some datasets are provided with different space grid. For the following processing, you might need to change the path of the source and target folder. 

For PDEBench Compressible Navier-Stokes dataset, use `convert_com_ns.py` to downsample the a subset of the dataset from 512x512 to 128x128. 

For CFDBench dataset, use `convert_cfdbench.py` to upsample the dataset from 64x64 to 128x128 as well as performing train/val/test split.

## Step 3. Prepare configs.
Modify or create a new config file under the folder `configs/data` and specify the correct path to the folders and files. 