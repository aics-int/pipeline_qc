import numpy as np
import os

from tempfile import TemporaryDirectory
from typing import Dict
from datetime import datetime
from aicsimageio.writers import ome_tiff_writer
from pipeline_qc.image_qc_methods import file_processing_methods, query_fovs
from model_zoo_3d_segmentation.zoo import SegModel, SuperModel
from .cell_seg_uploader import CellSegmentationUploader

# Constants
MODEL = "DNA_MEM_instance_LF_integration_two_camera"

class CellSegmentationWrapper:
    """
    Single cell ML Segmentation wrapper
    Wraps the core segmentation code from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse
    and performs additional query and upload tasks for microscopy pipeline usage
    """

    def __init__(self, uploader: CellSegmentationUploader):
        if not uploader:
            raise AttributeError("uploader")
        self._uploader = uploader

    def single_seg_run(self, image):
        sm = SuperModel(MODEL)

        return sm.apply_on_single_zstack(input_img=image, inputCh=[0, 1, 2])

    def _create_segmentable_image(self, localfilepath, sourceimagefileid):

        channel_dict = file_processing_methods.split_image_into_channels(localfilepath, sourceimagefileid)

        full_im_list = list()
        for channel in ['405nm', '638nm', 'brightfield']:
            for key, value in channel_dict.items():
                if key == channel:
                    full_im_list.append(value)

        return np.array(full_im_list)

    def batch_cell_segmentations(self, workflows=None, cell_lines=None, plates=None, fovids=None,
                                 only_from_fms=True, save_to_fms=False, save_to_isilon=False,
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
                im = self._create_segmentable_image(row['localfilepath'], row['sourceimagefileid'])
                if im.shape[0] ==3:
                    comb_seg = self.single_seg_run(im)
                else:
                    print(f'FOV:{row["fovid"]} does not have nucleus or cellular color channels')
                    break

                file_name = f'{row["fovid"]}_cellSegCombined.ome.tiff'
 
                if save_to_fms == True:
                    print("Uploading output file to FMS")

                    with TemporaryDirectory() as tmp_dir:
                        local_file_path = f'{tmp_dir}/{file_name}'
                        with ome_tiff_writer.OmeTiffWriter(local_file_path) as writer:
                            writer.save(comb_seg)
                        self._uploader.upload_combined_segmentation(local_file_path, row["sourceimagefileid"])

                if save_to_isilon == True:
                    print("Saving output file to Isilon")
                    with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{file_name}') as writer:
                        writer.save(comb_seg)

        return
