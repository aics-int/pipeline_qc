# Upload aligned H2B data to fms, with alignment info in metadata block
import os
import logging
import pandas as pd
import re
from copy import deepcopy

from aicsfiles import FileManagementSystem
from aicsfiles.filter import Filter
from lkaccess import LabKey

# Example row in the INPUT_CSV
# | change_median_intensity | coor_dist_qc | date    | diff_mse | dist_sum_diff | folder        | image_type | instrument | mse_qc | qc  | num_beads | num_beads_qc | rotate_angle | scaling    | shift_x   | shift_y    |
# | -1.306519255            | 1	           | 20190813| 7.46E-05	| 0.143271547   | ZSD3_20190813	| beads	     | ZSD3	      | 0      |pass | 32        |  1           | -0.002435847 | 0.999571786|1.228227663|-0.465022644|

log = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)


def _check_metadata(file):
    """
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
    """
    assert 'file' in file, '"file" block missing from metadata'
    assert 'file_id' in file['file'], '"file_id" missing from "file" metadata block'
    assert 'file_name' in file['file'], '"file_name" missing from "file" metadata block'
    assert 'original_path' in file['file'], '"original_path" missing from "file" metadata block'
    assert 'content_processing' in file, '"content_processing" block missing from metadata'
    assert 'microscopy' in file, '"microscopy" block missing from metadata'
    assert 'fov_id' in file['microscopy'], '"fov_id" missing from "microscopy" metadata block'


def _grab_zsd_and_date(file):
    file_name = file['file']['file_name']
    file_path = file['file']['original_path']
    zsd = 'ZSD' + file_path.split('ZSD')[1][0]
    # Grab the date from the file name, assuming it's in the format YYYYMMDD and bookended by underscores or hyphens
    date = re.search('[_-]([0-9]{8})[_-]', file_name).group(1)

    log.debug(f'Info for file {file_name}')
    log.debug(f'Path:       {file_path}')
    log.debug(f'Date :      {date}')
    log.debug(f'Instrument: {zsd}')

    return zsd, date


def _update_aligned_file_metadata(original_file, filtered_df):
    new_metadata = deepcopy(original_file)

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

    return new_metadata


def _update_failed_files(failed_files, file, folder, failure_msg):
    log.info(f'Failing file {file} ({failure_msg})')
    new_failed_files = failed_files.append({'FileName': file, 'Failure': failure_msg}, ignore_index=True)
    new_failed_files.to_csv(os.path.join(folder, 'fail_upload.csv'))
    return new_failed_files


def upload_aligned_files(lk: LabKey, input_csv: str, folder: str):
    fms = FileManagementSystem(lk_conn=lk)
    df = pd.read_csv(input_csv)

    # alignment_reference
    df['date'] = df['date'].astype(str)

    # String or Path object of file to be uploaded to FMS
    failed_files = pd.DataFrame(columns=['FileName', 'Failure'])
    all_files = os.listdir(folder)

    # TODO: The number of files this loop handles can be fairly high (Up to around ~2000). Would be worth parallelizing
    #       or approaching in a different manner.
    for file in all_files:
        if file.endswith('.tiff'):
            new_file_path = os.path.join(folder, file)

            # Find the file we want to copy the initial metadata blob from
            content_proc_filter = Filter().with_file_name(file.replace('-alignV2', '').replace('.tiff', '.ome.tiff'))

            fms_result = fms.query_files(content_proc_filter)
            original_file = fms_result[0]

            try:
                _check_metadata(original_file)
            except Exception as e:
                failed_files = _update_failed_files(failed_files, file, folder, str(e))
                continue

            # get instrument and date from file
            try:
                zsd, date = _grab_zsd_and_date(original_file)
            except Exception as e:
                failed_files = _update_failed_files(failed_files, file, folder, f"Issue grabbing ZSD or Date: {str(e)}")
                continue

            # Only upload files with 'pass' camera-alignment status for now
            filtered_df = df.loc[(df['instrument'] == zsd) & (df['date'] == date) & (df['qc'] == 'pass')]

            if len(filtered_df.index) > 0:
                log.info(f'Uploading file {file}')
                new_metadata = _update_aligned_file_metadata(original_file, filtered_df)

                aligned_file = fms.upload_file(new_file_path, new_metadata)

                fov_id = original_file['microscopy']['fov_id']
                lk.update_rows(
                    schema_name='microscopy',
                    query_name='FOV',
                    rows=[{'FovId': fov_id, 'AlignedImageFileId': aligned_file.file_id}]
                )
            else:
                failed_files = _update_failed_files(failed_files, file, folder,
                                                    'No corresponding file with "QC: Pass": found')
        else:
            failed_files = _update_failed_files(failed_files, file, folder, "Not a .tiff")
