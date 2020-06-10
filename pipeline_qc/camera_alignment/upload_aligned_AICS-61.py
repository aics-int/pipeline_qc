from pipeline_qc.camera_alignment import upload_aligned_files_to_fms
from lkaccess import contexts, LabKey

'''
This file exists because `upload_aligned_files_to_fms.py` used to be hard-coded to upload for AICS-61
'''

LK_ENV = contexts.STAGE  # The LabKey environment to use
# Example row in the INPUT_CSV
# | change_median_intensity | coor_dist_qc | date    | diff_mse | dist_sum_diff | folder        | image_type | instrument | mse_qc | qc  | num_beads | num_beads_qc | rotate_angle | scaling    | shift_x   | shift_y    |
# | -1.306519255            | 1	           | 20190813| 7.46E-05	| 0.143271547   | ZSD3_20190813	| beads	     | ZSD3	      | 0      |pass | 32        |  1           | -0.002435847 | 0.999571786|1.228227663|-0.465022644|
INPUT_CSV = '/allen/aics/microscopy/Data/alignV2/align_info.csv'
FOLDER = '/allen/aics/microscopy/Data/alignV2/AICS-61'


if __name__ == '__main__':
    lk = LabKey(server_context=LK_ENV)
    upload_aligned_files_to_fms.upload_aligned_files(lk, INPUT_CSV, FOLDER)
