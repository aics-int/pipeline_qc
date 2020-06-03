# Upload aligned H2B data to fms, with alignment info in metadata block
import os
import logging
import pandas as pd
from copy import deepcopy

from aicsfiles import FileManagementSystem
from aicsfiles.filter import Filter
from lkaccess import LabKey, contexts

LK_ENV = contexts.STAGE  # The LabKey environment to use
# Example row in the INPUT_CSV
# | change_median_intensity | coor_dist_qc | date    | diff_mse | dist_sum_diff | folder        | image_type | instrument | mse_qc | qc  | num_beads | num_beads_qc | rotate_angle | scaling    | shift_x   | shift_y    |
# | -1.306519255            | 1	           | 20190813| 7.46E-05	| 0.143271547   | ZSD3_20190813	| beads	     | ZSD3	      | 0      |pass | 32        |  1           | -0.002435847 | 0.999571786|1.228227663|-0.465022644|
INPUT_CSV = '/allen/aics/microscopy/Data/alignV2/align_info.csv'
FOLDER = '/allen/aics/microscopy/Data/alignV2/AICS-61'

log = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)


def upload_aligned_files(lk: LabKey, input_csv: str, folder: str):
    fms = FileManagementSystem(lk_conn=lk)
    df = pd.read_csv(input_csv)

    # alignment_reference
    df['date'] = df['date'].astype(str)

    # String or Path object of file to be uploaded to FMS
    failed_files = []
    all_files = os.listdir(folder)

    # TODO: The number of files this loop handles can be fairly high (Up to around ~2000). Would be worth parallelizing or
    #       approaching in a different manner.
    for file in all_files:
        if file.endswith('.tiff'):
            log.debug('')
            new_file_path = os.path.join(folder, file)

            # Find the file we want to copy the initial metadata blob from
            content_proc_filter = Filter().with_file_name(file.replace('-alignV2', '').replace('.tiff', '.ome.tiff'))
            '''
            The queried file is expected to have at least the following metadata:
            {
                file: {
                    file_id: '',
                    file_name: '',
                    original_path: '',
                }
                content_processing: {},
                microscopy: {
                    fov_id: ''
                }
            }
            '''
            fms_result = fms.query_files(content_proc_filter)
            original_file = fms_result[0]

            # get instrument and date from file
            file_path = original_file['file']['original_path']
            zsd = 'ZSD' + file_path.split('ZSD')[1][0]
            date = original_file['file']['file_name'].split('_')[2]

            # Only upload files with 'pass' camera-alignment status for now
            filtered_df = df.loc[(df['instrument'] == zsd) & (df['date'] == date) & (df['qc'] == 'pass')]
            if len(filtered_df.index) > 0:
                # Edit metadata accordingly
                new_metadata = deepcopy(original_file)  # This is a shallow copy, but it technically doesn't matter

                if 'content_processing' in original_file:
                    # get align info
                    shift_x = filtered_df['shift_x'].values[0]
                    shift_y = filtered_df['shift_y'].values[0]
                    scaling = filtered_df['scaling'].values[0]
                    rotation_angle = filtered_df['rotate_angle'].values[0]

                    new_metadata['content_processing']['two_camera_alignment'] = {}
                    new_metadata['content_processing']['two_camera_alignment']['algorithm_version'] = 'alignV2'
                    new_metadata['content_processing']['two_camera_alignment']['shift_x'] = shift_x
                    new_metadata['content_processing']['two_camera_alignment']['shift_y'] = shift_y
                    new_metadata['content_processing']['two_camera_alignment']['scaling'] = scaling
                    new_metadata['content_processing']['two_camera_alignment']['rotation_angle'] = rotation_angle

                new_metadata['provenance'] = {}
                new_metadata['provenance']['input_files'] = [original_file['file']['file_id']]
                new_metadata['provenance']['algorithm'] = 'OmeTiffCameraAlignment'

                aligned_file = fms.upload_file(new_file_path, new_metadata)

                fov_id = original_file['microscopy']['fov_id']
                lk.update_rows(
                    schema_name='microscopy',
                    query_name='FOV',
                    rows=[{'FovId': fov_id, 'AlignedImageFileId': aligned_file.file_id}]
                )
            else:
                failed_files.append(file)

    pd.DataFrame(failed_files).to_csv(os.path.join(folder, 'fail_upload.csv'))
