# Example script to batch process optical control plates based on cell line

import os
from pipeline_qc import camera_alignment
import lkaccess
import pandas as pd

# Query for H2B data
# Customize dataframe to date and zsd
# Crawl through all optical control plates
# If optical control plate doesn't contain psf.czi, go look for argolight, if that still doesnt exist, throw an error
# Set image path
# Run exe

# Connect to host on labkey
lk = lkaccess.LabKey(host="aics.corp.alleninstitute.org")

# Filter from FOV table, specify columns in dataframe
my_results = lk.select_rows_as_list(
    schema_name='microscopy',
    query_name='FOV',
    view_name='FOV -handoff',
    filter_array=[
        ('Objective', '100', 'eq'),
        ('SourceImageFileId/Filename', '100X', 'contains'),
        ('WellId/PlateId/Workflow/Name', '4.4', 'contains'),
        ('QCStatusId/Name', 'Passed', 'eq'),
        ('WellId/PlateId/PlateTypeId/Name', 'Production - Imaging', 'eq'),
        ('SourceImageFileId/CellLineId/Name', 'AICS-13', 'contains'),
        #('SourceImageFileId/Filename', '_aligned_cropped', 'contains')
    ],
    columns=[
        'FOVId',
        'FOVImageDate',
        'InstrumentId',
        'InstrumentId/Name',
        'Objective',
        'QCStatusId/Name',
        'SourceImageFileId',
        'SourceImageFileId/CellLineId/Name',
        'SourceImageFileId/CellPopulationId/Clone',
        'SourceImageFileId/Filename',
        'WellId',
        'WellId/PlateId',
        'WellId/PlateId/PlateTypeId/Name',
        'WellId/PlateId/Workflow/Name',
        'WellId/WellName/Name',
        'WellId/PlateId/BarCode',
        'SourceImageFileId/LocalFilePath'
    ]
)
df = pd.DataFrame(my_results)
df = pd.read_csv(r'\\allen\aics\microscopy\Calysta\projects\training_4_4_h2b\csvs\pipeline_alignv2_test.csv')
# Add imaging date to read path
for index, row in df.iterrows():
    file_name = row['SourceImageFileId/Filename']
    imaging_date = file_name.split('_')[2][0:8]
    df.loc[index, 'ImagingDate'] = imaging_date

df_date_zsd = df[['ImagingDate', 'InstrumentId/Name']]
df_date_zsd = df_date_zsd.drop_duplicates()
# df_date_zsd = df_date_zsd.drop(index=5) # Todo: Fix this entry on Labkey, date == 20193115

bead_production_path = r'\\allen\aics\microscopy\PRODUCTION\OpticalControl'
ring_production_path = r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ARGO-POWER'

df_date_zsd = pd.DataFrame(columns=['ImagingDate', 'InstrumentId/Name'])
#df_date_zsd = df_date_zsd.append({'ImagingDate': '20190809', 'InstrumentId/Name': 'ZSD-1'}, ignore_index=True)
#df_date_zsd = df_date_zsd.append({'ImagingDate': '20190813', 'InstrumentId/Name': 'ZSD-1'}, ignore_index=True)
#df_date_zsd = df_date_zsd.append({'ImagingDate': '20191112', 'InstrumentId/Name': 'ZSD-1'}, ignore_index=True)
df_date_zsd = df_date_zsd.append({'ImagingDate': '20190201', 'InstrumentId/Name': 'ZSD-2'}, ignore_index=True)

update_align_info = True

align_info = r'\\allen\aics\microscopy\Data\alignV2\align_info.csv'
df_align_info = pd.read_csv(align_info)

for index, row in df_date_zsd.iterrows():
    file_path = None
    system = row['InstrumentId/Name'][-1]
    date = row['ImagingDate']
    print('processing ZSD' + system + ' on ' + date)
    # Optical Control Plate path
    optical_control_plate_path = os.path.join(bead_production_path, 'ZSD' + str(system) + '_' + date)
    optical_control_plate_exists = os.path.isdir(optical_control_plate_path)

    count = 0
    if optical_control_plate_exists:
        sim_matrix = False
        for file in os.listdir(optical_control_plate_path):
            if file.endswith('sim_matrix.txt'):
                sim_matrix = True
                count = 1
                filepath = None

        if not sim_matrix:
            print('reading: '+ optical_control_plate_path)
            plate_files = os.listdir(optical_control_plate_path)

            for file in plate_files:
                if file.endswith('psf.czi'):
                    filepath = os.path.join(optical_control_plate_path, file)
                    image_type = 'beads'
                    count += 1
                if count > 1:
                    print('Error: multiple psf.czi files')

    if count == 0:
        print('No psf.czi found')
        # Find corresponding argolight image
        argo_system_path = os.path.join(ring_production_path, 'ZSD' + str(system), 'split_scenes', str(date))

        # check if directory exists
        print('finding ' + argo_system_path)
        if os.path.isdir(argo_system_path):
            sim_matrix = False
            all_argo_imgs = os.listdir(argo_system_path)
            for argo_img in all_argo_imgs:
                if argo_img.endswith('sim_matrix.txt'):
                    sim_matrix = True
                    count = 1
                    filepath = None

            if not sim_matrix:
                for argo_img in all_argo_imgs:
                    if argo_img.endswith('P3.czi'):
                        filepath = os.path.join(argo_system_path, argo_img)
                        image_type = 'rings'
                        count += 1
                if count > 1:
                    print('Error: multiple argo files')
                if count == 0:
                    print('Error: no argo img found')
        else:
            print('Error: no argo folder found')
            filepath = None

    if filepath is not None:
        print('aligning: ' + filepath)
        exe = camera_alignment.Executor(
            image_path=filepath,
            image_type=image_type,
            ref_channel_index='EGFP',
            mov_channel_index='CMDRP',
            system_type='zsd',
            thresh_488=None,  # Set 'None' to use default setting
            thresh_638=None,  # Set 'None' to use default setting
            crop_center=None,  # Set 'None' to use default setting
            method_logging=True,
            align_mov_img=True,
            align_mov_img_path=filepath,
            align_mov_img_file_extension='_aligned.tif',
            align_matrix_file_extension='_sim_matrix.txt')

        transformation_parameters_dict, bead_num_qc, num_beads, changes_fov_intensity_dictionary,\
            coor_dist_qc, diff_sum_beads, mse_qc, diff_mse = exe.execute()

        if update_align_info:
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
                   'diff_mse': diff_mse
                   }
            df_align_info = df_align_info.append(row, ignore_index=True)

df_align_info.to_csv(r'\\allen\aics\microscopy\Data\alignV2\updated.csv')