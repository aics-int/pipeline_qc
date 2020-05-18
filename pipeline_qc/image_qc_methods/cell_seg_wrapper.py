import numpy as np
import os
from pipeline_qc.image_qc_methods import file_processing_methods, query_fovs
from aicsimageio.writers import ome_tiff_writer
from model_zoo_3d_segmentation.zoo import SegModel, SuperModel

MODEL = "DNA_MEM_instance_LF_integration_two_camera"


def single_seg_run(image):
    sm = SuperModel(MODEL)

    return sm.apply_on_single_zstack(input_img=image, inputCh=[0, 1, 2])


def create_segmentable_image(localfilepath, sourceimagefileid):

    channel_dict = file_processing_methods.split_image_into_channels(localfilepath, sourceimagefileid)

    full_im_list = list()
    for channel in ['405nm', '638nm', 'brightfield']:
        for key, value in channel_dict.items():
            if key == channel:
                full_im_list.append(value)

    return np.array(full_im_list)


def batch_cell_segmentations(workflows=None, cell_lines=None, plates=None, fovids=None,
                             only_from_fms=True, save_to_fms=False, save_to_isilon=True,
                             output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'):
    query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                     only_from_fms=only_from_fms)

    print(f'''
    __________________________________________

    {len(query_df)} fovs were found to process.

    __________________________________________
    ''')

    for index, row in query_df.iterrows():

        if os.path.isfile(f'{output_dir}/{row["fovid"]}.ome.tif'):
            print(f'FOV:{row["fovid"]} has already been segmented')
        else:
            print(f'Running Segmentation on fov:{row["fovid"]}')
            im = create_segmentable_image(row['localfilepath'], row['sourceimagefileid'])
            comb_seg = single_seg_run(im)
            if save_to_fms == True:
                # UPLOAD TO FMS GOES HERE
            if save_to_isilon == True:
                with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{row["fovid"]}.ome.tif') as writer:
                    writer.save(comb_seg)

    return