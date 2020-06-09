import numpy as np
import os

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from datetime import datetime
from aicsimageio.writers import ome_tiff_writer
from pipeline_qc.image_qc_methods import file_processing_methods, query_fovs
from model_zoo_3d_segmentation.zoo import SegModel, SuperModel
from pipeline_qc.image_qc_methods.cell_seg_repository import CellSegmentationRepository
from labkey.utils import ServerContext

# Constants
MODEL = "DNA_MEM_instance_LF_integration_two_camera"

class CellSegmentationWrapper:
    """
    Single cell ML Segmentation wrapper
    Wraps the core segmentation code from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse
    and performs additional query and upload tasks for microscopy pipeline usage
    """

    def __init__(self, repository: CellSegmentationRepository, labkey_context: ServerContext):
        if repository is None:
            raise AttributeError("repository")
        self._repository = repository
        self._labkey_context = labkey_context

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
                                 only_from_fms=True, save_to_fms=False, save_to_isilon=False, process_duplicates=False,
                                 output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'): 
        """
        Process segmentations as a batch. 
        FOV images are queried from FMS based on the given options and segmented sequentially.
        """                                
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                        only_from_fms=only_from_fms)


        print(f'''
        __________________________________________

        {len(query_df)} fovs were found to process.

        __________________________________________
        ''')

        for index, row in query_df.iterrows():
            file_name = self._get_seg_filename(row['localfilepath'])

            if not process_duplicates and self._repository.segmentation_exists(file_name):
                print(f'FOV:{row["fovid"]} has already been segmented') 
                continue
            
            print(f'Running Segmentation on fov:{row["fovid"]}')
            im = self._create_segmentable_image(row['localfilepath'], row['sourceimagefileid'])
            if im.shape[0] ==3:
                comb_seg = self.single_seg_run(im)
            else:
                print(f'FOV:{row["fovid"]} does not have nucleus or cellular color channels')
                continue
            
            if save_to_fms == True:
                print("Uploading output file to FMS")

                with TemporaryDirectory() as tmp_dir:
                    local_file_path = f'{tmp_dir}/{file_name}'
                    with ome_tiff_writer.OmeTiffWriter(local_file_path) as writer:
                        writer.save(comb_seg)
                    self._repository.upload_combined_segmentation(local_file_path, row["sourceimagefileid"])

            if save_to_isilon == True:
                print("Saving output file to Isilon")
                with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{file_name}') as writer:
                    writer.save(comb_seg)

        return

    def _get_seg_filename(self, fov_file_path: str):
        """
        Generate appropriate segmentation filename based on FOV file name
        Will look like this: {barcode}-{obj}-{date}-{colony pos(optional)}-{scene}-{pos}-{well}_CellNucSegCombined.ome.tiff
        """
        if fov_file_path.endswith(".ome.tiff"):
            file_prefix = Path(fov_file_path[:-9]).stem
        else:
            file_prefix = Path(fov_file_path).stem
        
        file_prefix = file_prefix.replace("-alignV2", "").replace("alignV2", "") # get rid of alignV2 in all its forms
        return f"{file_prefix}_CellNucSegCombined.ome.tiff"