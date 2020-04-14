# Upload aligned H2B data to fms, with alignment info in metadata block

from aicsfiles import FileManagementSystem
from aicsfiles.filter import Filter
import os
import pandas as pd

# Create an FMS connection (to stg-aics when testing!!)
prod_fms = FileManagementSystem(host='aics')

# alignment_reference
df = pd.read_csv('/allen/aics/microscopy/Data/alignV2/align_info.csv')
df['date'] = df['date'].astype(str)

# String or Path object of file to be uploaded to FMS
folder = '/allen/aics/microscopy/Calysta/test/h2b_aligned'
failed_files = []
all_files = os.listdir(failed_files)
for file in all_files:
    if file.endswith('.tiff'):
        #file = '3500002669_100X_20190118_1-alignV2-Scene-54-P55-C08.tiff'
        new_file_path = os.path.join(folder, file)

        # Find the file we want to copy the initial metadata blob from
        content_proc_filter = Filter().with_file_name(file.replace('-alignV2', '').replace('.tiff', '.czi'))
        result = prod_fms.query_files(content_proc_filter)

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
                prod_fms.upload_file(new_file_path, new_metadata)
        else:
            failed_files.append(file)

pd.DataFrame(failed_files).to_csv(os.path.join(folder, 'fail_upload.csv'))