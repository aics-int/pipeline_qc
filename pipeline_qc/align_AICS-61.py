from skimage import io, transform as tf
from aicsimageio import AICSImage, writers
import os
import pandas as pd
import numpy as np
from lkaccess import LabKey

def perform_similarity_matrix_transform(img, matrix):
    """
    Performs a similarity matrix geometric transform on an image
    :param img: A 2D/3D image to be transformed
    :param matrix: Similarity matrix to be applied on the image
    :param output_path: Output path to save the image
    :param filename: Name of the image to save
    :return:
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
        # io.imsave(output_path, after_transform)

    return after_transform

def generate_augments(img, path, img_size=(360, 536)):
    y_size, x_size = img_size
    flippedlr = np.zeros(img.shape)
    flippedud = np.zeros(img.shape)
    rot180 = np.zeros(img.shape)

    for z in range(0, img.shape[0]):
        z_slice = img[z, :, :]
        flippedlr[z, :, :] = np.fliplr(z_slice).astype(np.uint16)
        flippedud[z, :, :] = np.flipud(z_slice).astype(np.uint16)
        rot180[z, :, :] = tf.rotate(z_slice, angle=180., order=3, preserve_range=True).astype(np.uint16)
    row = {}
    for key, crop_4x_img in {'_cropped': img,
                             '_flippedlr': flippedlr,
                             '_flippedud': flippedud,
                             '_rot180': rot180}.items():
        final_crop = crop_4x_img[:,
                     int(crop_4x_img.shape[1] / 2 - y_size / 2):int(crop_4x_img.shape[1] / 2 + y_size / 2),
                     int(crop_4x_img.shape[2] / 2 - x_size / 2):int(crop_4x_img.shape[2] / 2 + x_size / 2)]

        io.imsave(path.replace('.ome', key + '.ome'), final_crop)
        row.update({key[1:]: path.replace('.ome', key + '.ome')})
    return row

lk = LabKey(host="aics.corp.alleninstitute.org")

def get_query_per_cell_line(pipeline, cell_line):
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

pipeline = 'Pipeline 4.4'
cell_line = 'AICS-61'
output_folder = '/allen/aics/microscopy/Data/alignV2/' + cell_line
results = get_query_per_cell_line(pipeline, cell_line)
df = pd.DataFrame(results)

missing_folders = []
missing_files = []
processed_fov = []
failed_fov = []
optical_control = '/allen/aics/microscopy/PRODUCTION/OpticalControl'

for index, row in df.iterrows():
    print('processing: ' + str(index) + ' out of ' + str(len(df)))
    align_file = None
    tf_array = None
    image_date = str(row['SourceImageFileId/Filename'].split('_')[2][0:8])
    system = row['InstrumentId/Name'].replace('-', '')

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
        # Find it in argolight
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

    if path_exists and tf_array is not None:
        # read image
        file_name = row['SourceImageFileId/Filename'].replace('_aligned_cropped', '').split('.')[0]
        raw_split_file =row['SourceImageFileId/LocalFilePath']

        if raw_split_file is not None:
            img_data = AICSImage(raw_split_file)
            channels = img_data.get_channel_names()
            img_stack = img_data.data
            omexml = img_data.metadata
            # process each channel
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

                # save out for label free
                # io.imsave(row[sub_folder], img)

                # generate stack for data back fill
                final_img[0, 0, channels.index(channel), :, :, :] = img



                # generate augments, save out for label free
                # augment_row = generate_augments(img=img, path=row[sub_folder], img_size=(360, 536))

            pix = omexml.image().Pixels
            pix.set_SizeX(900)
            pix.set_SizeY(600)

            final_img = final_img.astype(np.uint16)
            upload_img = final_img[0, :, :, :, 12:612, 12:912]
            upload_img = upload_img.transpose((0, 2, 1, 3, 4))

            writer = writers.OmeTiffWriter(os.path.join(output_folder, file_name.replace('-Scene', '-alignV2-Scene') + '.tiff'))
            writer.save(upload_img.astype(np.uint16),
                        ome_xml=omexml)
                        #channel_names=channels,
                        #pixels_physical_size=img_data.get_physical_pixel_size(),
                        #dimension_order="STCZYX"


            # io.imsave(os.path.join(output_folder, file_name.replace('-Scene', '-alignV2-Scene') + '.tiff'), upload_img.astype(np.uint16))
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