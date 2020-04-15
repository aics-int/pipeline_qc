# Upload aligned H2B data to fms, with alignment info in metadata block
from aicsfiles import FileManagementSystem
from aicsfiles.filter import Filter
import os
import pandas as pd

LK_ENV = 'stg-aics'
INPUT_CSV = '/allen/aics/microscopy/Data/alignV2/align_info.csv'
FOLDER = '/allen/aics/microscopy/Data/alignV2/AICS-61'

fms = FileManagementSystem(host=LK_ENV)
df = pd.read_csv(INPUT_CSV)

# alignment_reference
df['date'] = df['date'].astype(str)

# String or Path object of file to be uploaded to FMS
failed_files = []
all_files = os.listdir(FOLDER)
for file in all_files:
    if file.endswith('.tiff'):
        new_file_path = os.path.join(FOLDER, file)

        # Find the file we want to copy the initial metadata blob from
        content_proc_filter = Filter().with_file_name(file.replace('-alignV2', '').replace('.tiff', '.czi'))
        result = fms.query_files(content_proc_filter)

        # get instrument and date from file
        file_path = result[0]['file']['original_path']
        zsd = 'ZSD' + file_path.split('ZSD')[1][0]
        date = result[0]['file']['file_name'].split('_')[2]

        # Only upload files with 'pass' camera-alignment status for now
        if df.loc[(df['instrument'] == zsd) & (df['date'] == date), 'qc'] == 'pass':
            # get align info
            shift_x = df.loc[(df['instrument'] == zsd) & (df['date'] == date), 'shift_x'].values.tolist()[0]
            shift_y = df.loc[(df['instrument'] == zsd) & (df['date'] == date), 'shift_y'].values.tolist()[0]
            scaling = df.loc[(df['instrument'] == zsd) & (df['date'] == date), 'scaling'].values.tolist()[0]
            rotation_angle = df.loc[(df['instrument'] == zsd) & (df['date'] == date), 'rotate_angle'].values.tolist()[0]

            if 'content_processing' in result[0]:
                new_metadata = result[0]
                # Edit metadata accordingly
                new_metadata['content_processing']['two_camera_alignment'] = {}
                new_metadata['content_processing']['two_camera_alignment']['algorithm_version'] = 'alignV2'
                new_metadata['content_processing']['two_camera_alignment']['shift_x'] = shift_x
                new_metadata['content_processing']['two_camera_alignment']['shift_y'] = shift_y
                new_metadata['content_processing']['two_camera_alignment']['scaling'] = scaling
                new_metadata['content_processing']['two_camera_alignment']['rotation_angle'] = rotation_angle
                fms.upload_file(new_file_path, new_metadata)
        else:
            failed_files.append(file)

pd.DataFrame(failed_files).to_csv(os.path.join(FOLDER, 'fail_upload.csv'))