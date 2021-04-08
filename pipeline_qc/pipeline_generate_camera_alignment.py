# -*- coding: utf-8 -*-
"""
This script is used to generate the camera alignment transformation matrix.
The script takes in an optical control image (e.g. argolight rings) and generate a
similarity transform matrix, and store the aligned ring image (for visual inspection)
and parameters and qc status to a csv (align_info_2021.csv) on the isilon
Instructions to use:
    1. Prepare the argolight images by splitting them using Zen Blue
    2. Edit the file path (line 18) for the variable 'optical_control_img_filepath' to point to the ring image
    3. Edit the system (line 20) and date (line 21) infromation
    4. Hit run
    5. Check on the Console (on the right panel) and see if the transform looks good
    (6. If transform looks bad, open the aligned image and compare with the EGFP 
     channel image to check the alignment quality)
    
"""

## User edits here:
optical_control_img_filepath = r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ARGO-POWER\ZSD1\split_scenes\20210222\argo_100X_20210219_P3.czi'
system = 'ZSD1'
date = '20210222'

#===================================
## Core script
import os
from pipeline_qc import obtain_camera_alignment
from pipeline_qc.camera_alignment.apply_camera_alignment_utilities import perform_similarity_matrix_transform
import pandas as pd

image_type = 'rings'
ref_channel = 'EGFP'
mov_channel = 'CMDRP'
system_type = 'zsd'

align_info = r'\\allen\aics\microscopy\Data\alignV2\align_info_2021.csv'
df_align_info = pd.read_csv(align_info)

exe = obtain_camera_alignment.Executor(
    image_path=optical_control_img_filepath,
    image_type=image_type,
    ref_channel_index=ref_channel,
    mov_channel_index=mov_channel,
    system_type=system_type,
    thresh_488=None,  # Set 'None' to use default setting
    thresh_638=None,  # Set 'None' to use default setting
    ref_seg_param=None, # Set 'None' to use default setting
    mov_seg_param=None, # Set 'None' to use default setting
    crop_center=None,  # Set 'None' to use default setting
    method_logging=True,
    align_mov_img=True,
    align_mov_img_path=optical_control_img_filepath,
    align_mov_img_file_extension='_aligned.tif',
    align_matrix_file_extension='_sim_matrix.txt')  
    
transformation_parameters_dict, bead_num_qc, num_beads, changes_fov_intensity_dictionary,\
coor_dist_qc, diff_sum_beads, mse_qc, diff_mse, z_offset, ref_signal, ref_noise, mov_signal, mov_noise = exe.execute()

if bead_num_qc & coor_dist_qc & mse_qc:
    qc = 'pass'
else:
    qc = 'fail'

row = {'folder': system,
       'instrument': system,
       'date': date,
       'image_type': image_type,
       'shift_x': transformation_parameters_dict['shift_x'],
       'shift_y': transformation_parameters_dict['shift_y'],
       'rotate_angle': transformation_parameters_dict['rotate_angle'],
       'scaling': transformation_parameters_dict['scaling'],
       'num_beads': num_beads,
       'num_beads_qc': bead_num_qc,
       'change_median_intensity': changes_fov_intensity_dictionary['median_intensity'],
       'coor_dist_qc': coor_dist_qc,
       'dist_sum_diff': diff_sum_beads,
       'mse_qc': mse_qc,
       'diff_mse': diff_mse,
       'qc': qc
       }

df_align_info = df_align_info.append(row, ignore_index=True)

df_align_info = df_align_info[
    ['folder', 'instrument', 'date', 'image_type', 'shift_x', 'shift_y', 'rotate_angle', 
     'scaling', 'num_beads', 'change_median_intensity', 'coor_dist_qc', 'dist_sum_diff', 
     'mse_qc', 'diff_mse', 'qc']
]

df_align_info.to_csv(r'\\allen\aics\microscopy\Data\alignV2\align_info_2021.csv')