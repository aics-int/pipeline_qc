from skimage import transform as tf
from aicsimageio import AICSImage, writers
import os
import pandas as pd
import numpy as np
from tifffile import TiffFile
from ome_types import from_xml, to_xml
from cellbrowser_tools.fov_processing import _clean_ome_xml_for_known_issues
from lkaccess import LabKey

def perform_similarity_matrix_transform(img, matrix):
    """
    Performs a similarity matrix geometric transform on an image
    Parameters
    ----------
    img
        A 2D/3D image to be transformed
    matrix
        Similarity matrix to be applied on the image
    Returns
    -------

    """
    after_transform = None
    if len(img.shape) == 2:
        after_transform = tf.warp(img, inverse_map=matrix, order=3)
    elif len(img.shape) == 3:
        after_transform = np.zeros(img.shape)
        for z in range(0, after_transform.shape[0]):
            after_transform[z, :, :] = tf.warp(img[z, :, :], inverse_map=matrix, order=3)
    else:
        print('dimensions invalid for img')

    if after_transform is not None:
        after_transform = (after_transform*65535).astype(np.uint16)

    return after_transform

lk = LabKey(host="aics.corp.alleninstitute.org")

def get_query_per_cell_line(pipeline, cell_line):
    """
    Query from labkey per cell line
    Parameters
    ----------
    pipeline
        Workflow name, e.g. 'Pipeline 4.4
    cell_line
        Cell line name, e.g. 'AICS-61'

    Returns
    -------

    """
    query_results = lk.select_rows_as_list(
        schema_name='microscopy',
        query_name='FOV',
        view_name='FOV -handoff',
        filter_array=[
            ('Objective', '100', 'eq'),
            ('SourceImageFileId/Filename', '100X', 'contains'),
            ('WellId/PlateId/Workflow/Name', pipeline, 'eq'),
            ('WellId/PlateId/PlateTypeId/Name', 'Production - Imaging', 'eq'),
            ('SourceImageFileId/CellLineId/Name', cell_line, 'contains'),
            ('SourceImageFileId/Filename', 'aligned_cropped', 'doesnotcontain')
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
    return query_results

# Define query parameters
pipeline = 'Pipeline 4.4'
cell_line = 'AICS-61'
output_folder = '/allen/aics/microscopy/Data/alignV2/' + cell_line + '_2'

# get query results per cell line from labkey
results = get_query_per_cell_line(pipeline, cell_line)
df = pd.DataFrame(results)

missing_folders = []
missing_files = []
processed_fov = []
failed_fov = []
optical_control = '\\' + '/allen/aics/microscopy/PRODUCTION/OpticalControl'.replace('/', '\\')

for index, row in df.iterrows():
    print('processing: ' + str(index) + ' out of ' + str(len(df)))

    # Initialize some variables
    align_file = None
    tf_array = None
    image_date = str(row['SourceImageFileId/Filename'].split('_')[2][0:8])
    system = row['InstrumentId/Name'].replace('-', '')

    # Find transformation array from either beads or argolight rings images

    # Find transformation array from beads optical control images
    optical_control_path = os.path.join(optical_control, system + '_' + image_date)
    path_exists = os.path.isdir(optical_control_path)
    if path_exists:
        optical_control_files = os.listdir(optical_control_path)
        for file in optical_control_files:
            if file.endswith('sim_matrix.txt'):
                align_file = file

        if align_file is not None:
            tf_array = np.loadtxt(os.path.join(optical_control_path, align_file), delimiter=',')
        else:
            print('cannot find file in plate: ' + system + '_' + image_date)
            missing_files.append(system + '_' + image_date)

    if align_file is None:
        # Find transformation array from argolight rings optical control images
        argo_path = os.path.join(optical_control, 'ARGO-POWER', system, 'split_scenes', image_date)
        argo_exists = os.path.isdir(argo_path)
        if argo_exists:
            optical_control_files = os.listdir(argo_path)
            for file in optical_control_files:
                if file.endswith('sim_matrix.txt'):
                    align_file = file
            if align_file is not None:
                tf_array = np.loadtxt(os.path.join(argo_path, align_file), delimiter=',')
            else:
                print('cannot find file in argo: ' + system + '_' + image_date)
                missing_files.append(system + '_' + image_date)

    # With raw image and transformation array, apply transformation matrix to align images, and save out as tiffs
    if path_exists and tf_array is not None:
        # read image
        file_name = row['SourceImageFileId/Filename'].replace('_aligned_cropped', '').split('.')[0]
        raw_split_file =row['SourceImageFileId/LocalFilePath']

        if raw_split_file is not None:
            # Read image
            img_data = AICSImage(raw_split_file)
            channels = img_data.get_channel_names()
            img_stack = img_data.data

            # Read omexml from image
            with TiffFile(raw_split_file) as tiff:
                if tiff.is_ome:
                    description = tiff.pages[0].description.strip()
                    description = _clean_ome_xml_for_known_issues(description)
                    # print(description)
                    omexml = from_xml(description)
                else:
                    # this is REALLY catastrophic. Its not expected to happen for AICS data.
                    raise ValueError("Bad OME TIFF file")

            # process each channel, align the channel if channel name starts with 'Bright' 'CMDR'
            final_img = np.zeros(img_stack.shape)
            for channel in channels:
                if channel.startswith('Bright'):
                    sub_folder = 'aligned_bf'
                elif channel.startswith('CMDR'):
                    sub_folder = 'aligned_cmdr'
                elif channel.startswith('H334'):
                    sub_folder = 'raw_nuc'
                elif channel.startswith('EGF'):
                    sub_folder = 'raw_gfp'

                img = img_stack[0, 0, channels.index(channel), :, :, :]
                if channel.startswith('Bright') or channel.startswith('CMDR'):
                    img = perform_similarity_matrix_transform(img, tf_array)

                # generate stack for data back fill
                final_img[0, 0, channels.index(channel), :, :, :] = img

            # Crop image to final dimension (600, 900)
            final_img = final_img.astype(np.uint16)
            upload_img = final_img[0, :, :, :, 12:612, 12:912]
            upload_img = upload_img.transpose((0, 2, 1, 3, 4))

            # Update omexml to include accurate pixel sizes in x and y
            p = omexml.images[0].pixels
            p.size_x = 900
            p.size_y = 600

            # Set ometif directory to save out the image
            ometif_dir = os.path.join('\\' + output_folder.replace('/', '\\'), file_name.replace('-Scene', '-alignV2-Scene') + '.tiff')

            # convert omexml from string to xml
            ome_str = to_xml(omexml)

            # appease ChimeraX and possibly others who expect to see this
            ome_str = '<?xml version="1.0" encoding="UTF-8"?>' + ome_str
            with writers.OmeTiffWriter(file_path=ometif_dir, overwrite_file=True) as writer:
                writer.save(
                    upload_img.astype(np.uint16),
                    ome_xml=ome_str,
                )

            processed_fov.append(row['FOVId'])

        else:
            print ('error in ' + str(row['FOVId']))
            failed_fov.append(row['FOVId'])
    else:
        print('cannot find folder in ' + system + '_' + image_date)
        missing_folders.append(system + '_' + image_date)

pd.DataFrame(missing_folders).to_csv(os.path.join(output_folder, 'missing_optical_control.csv'))
pd.DataFrame(missing_files).to_csv(os.path.join(output_folder, 'missing_align_file.csv'))
pd.DataFrame(processed_fov).to_csv(os.path.join(output_folder, 'processed_fov.csv'))
pd.DataFrame(failed_fov).to_csv(os.path.join(output_folder, 'failed_fov.csv'))