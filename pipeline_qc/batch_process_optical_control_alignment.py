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
    schema_name = 'microscopy',
    query_name='FOV-Well cell line  & Solution',
    view_name = 'Drug Perturbation Project',
    filter_array = [
        ('WorkflowId/Name', 'Pipeline 4.4', 'eq'),
        ('FileId/Filename', '.ome', 'contains'),
        ('FileId/Filename', '100X', 'contains'),
        ('FOVId/SourceImageFileId/CelllineId/Name', '61', 'contains'),
        ('FOVId/FOVImageDate', '2019-07-15', 'datelt'),
        ('FOVId/FOVImageDate', '2019-01-19', 'dategt'),
        ('FOVId/QCStatusId/Name', 'Passed', 'eq'),
    ],
    columns = [
        'FOVId',
        'FOVId/InstrumentId',
        'FOVId/QCStatusId',
        'FOVId/SourceImageFileId/CellLineId/Name',
        'FOVId/SourceImageFileId/CellPopulationId/Clone',
        'FileId',
        'FileId/Filename',
        'PlateId',
        'PlateId/PlateStatusId',
        'PlateId/PlateTypeId',
        'PlateLayoutId',
        'SolutionLotId/Concentration',
        'SolutionLotId/SolutionId/Name',
        'WellName',
        'WindowsReadPath',
        'WorkflowId',
        'FOVId/InstrumentId/Name',  # Specify InstrumentId name for system
        'PlateId/BarCode',             # Specify PlateId name to get actual barcode
        'WorkflowId/Name',          # Specify WorkflowId name to get pipeline
    ]
)
df = pd.DataFrame(my_results)

# Add imaging date to read path
for index, row in df.iterrows():
    file_name = row['FileId/Filename']
    imaging_date = file_name.split('_')[2][0:8]
    df.loc[index, 'ImagingDate'] = imaging_date

df_date_zsd = df[['ImagingDate', 'FOVId/InstrumentId/Name']]
df_date_zsd = df_date_zsd.drop_duplicates()
df_date_zsd = df_date_zsd.drop(index=3336) # Todo: Fix this entry on Labkey, date == 20193115

bead_production_path = r'\\allen\aics\microscopy\PRODUCTION\OpticalControl'
ring_production_path = r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ARGO-POWER'

for index, row in df_date_zsd.iterrows():
    file_path = None
    system = row['FOVId/InstrumentId/Name'][-1]
    date = row['ImagingDate']

    # Optical Control Plate path
    optical_control_plate_path = os.path.join(bead_production_path, 'ZSD' + str(system) + '_' + date)
    print('reading: '+ optical_control_plate_path)
    plate_files = os.listdir(optical_control_plate_path)
    count=0
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
        argo_system_path = os.path.join(ring_production_path, 'ZSD' + str(system))
        all_argo_imgs = os.listdir(argo_system_path)
        for argo_img in all_argo_imgs:
            if argo_img.endswith(str(date) + '.czi'):
                filepath = os.path.join(ring_production_path, 'ZSD' + str(system), argo_img)
                image_type = 'rings'
                count += 1
            if count > 1:
                print('Error: multiple argo files')
            if count == 0:
                print('Error: no argo img found')

    if filepath is not None:
        print ('aligning: ' + filepath)
        exe = camera_alignment.Executor(image_path=filepath,
                                        image_type=image_type,
                                        ref_channel_index='EGFP',
                                        mov_channel_index='CMDRP',
                                        bead_488_lower_thresh=99.4,
                                        bead_638_lower_thresh=99,
                                        method_logging=True,
                                        align_mov_img=True,
                                        align_mov_img_path=filepath,
                                        align_mov_img_file_extension='_aligned.tif',
                                        align_matrix_file_extension='_sim_matrix.txt')
        exe.execute()
