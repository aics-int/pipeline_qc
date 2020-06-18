# Single cell segmentation
A pipeline friendly wrapper for Single Cell ML Segmentation. This module wraps the [Model zoo 3D Segmentation code](https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse?at=refs%2Ftags%2F0.1.0-pipeline-compatible) which runs on GPU processing.

## Local setup
### Prerequisites
* Nvidia CUDA compatible GPU
* Nvidia CUDA 10.1+ Toolkit - [Download & Setup instructions](https://developer.nvidia.com/cuda-downloads)

### Setup
```
make install-local
```
A conda environment will be created under `venv/local`

## GPU Node (HPC) setup
This code is memory and compute intensive and requires a compatible GPU to run. It is recommended to be run on a HPC cluster (ie. Slurm) GPU node. This repository relies on C code that **must be compiled** on the corresponding GPU node that will be used to run the code. Our cluster currently supports four GPU types (`titanx`, `titanxp`, `gtx1080` and `V100`), please refer to [Using the Slurm Cluster](http://confluence.corp.alleninstitute.org/display/SF/Using+the+SLURM+Cluster#UsingtheSLURMCluster-GPUUsage) for instructions on how to request and use cluster GPU resources. 

To create a GPU compatible virtual environment, run one of the following commands from a compatible GPU node.

### Setup for Titanx / Titanxp / GTX1080 (Titan compatible)
```
make install-titan
```
A Titan compatible conda environment will be created under `venv/titan`

### Setup for V100
```
make install-v100
```
A V100 compatible conda environment will be created under `venv/v100`