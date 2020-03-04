#!/bin/bash

#SBATCH --job-name=pipeline_qc_test
#SBATCH --partition aics_gpu_general
#SBATCH --output /allen/aics/microscopy/Aditya/image_qc_outputs/output+files_slurm/$2.out
#SBATCH --gres gpu:v100:1

#SBATCH --array 0-29%100

ITEM_ARRAY=(
AICS-11
AICS-12
AICS-0
AICS-13
AICS-17
AICS-10
AICS-16
AICS-25
AICS-24
AICS-23
AICS-22
AICS-14
AICS-32
AICS-53
AICS-54
AICS-7
AICS-5
AICS-57
AICS-58
AICS-36
AICS-43
AICS-33
AICS-40
AICS-30
AICS-61
AICS-69
AICS-46
AICS-74
AICS-68
)



srun /allen/aics/microscopy/venvs/default/v0.1.x/bin/python /allen/aics/microscopy/Aditya/image_qc_outputs/main_test.py --output_dir /allen/aics/microscopy/Aditya/image_qc_outputs --cell_line ${ITEM_ARRAY[$SLURM_ARRAY_TASK_ID]}
