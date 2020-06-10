import numpy as np
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from datetime import datetime
from aicsimageio.writers import ome_tiff_writer
from pipeline_qc.image_qc_methods import file_processing_methods, query_fovs
from model_zoo_3d_segmentation.zoo import SegModel, SuperModel
from pipeline_qc.image_qc_methods.cell_seg_uploader import CellSegmentationUploader
from labkey.utils import ServerContext

# Constants
MODEL = "DNA_MEM_instance_LF_integration_two_camera"

class CellSegmentationWrapper:
    """
    Single cell ML Segmentation wrapper
    Wraps the core segmentation code from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse
    and performs additional query and upload tasks for microscopy pipeline usage
    """

    def __init__(self, uploader: CellSegmentationUploader, labkey_context: ServerContext):
        if not uploader:
            raise AttributeError("uploader")
        self._uploader = uploader
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
                                 only_from_fms=True, save_to_fms=False, save_to_isilon=False,
                                 output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'):
        print(f"** START batch_cell_segmentations: {datetime.now()}")
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                        only_from_fms=only_from_fms, labkey_context=self._labkey_context)


        print(f'''
        __________________________________________

        {len(query_df)} fovs were found to process.

        __________________________________________
        ''')

        for index, row in query_df.iterrows():
            file_name = self._get_seg_filename(row['localfilepath'])
                
            if os.path.isfile(f'{output_dir}/{file_name}'):
                print(f'FOV:{row["fovid"]} has already been segmented')
            else:
                print(f'Running Segmentation on fov:{row["fovid"]}')
                print(f"** START _create_segmentable_image: {datetime.now()}")
                im = self._create_segmentable_image(row['localfilepath'], row['sourceimagefileid'])
                print(f"** END _create_segmentable_image: {datetime.now()}")
                if im.shape[0] ==3:
                    print(f"** START single_seg_run: {datetime.now()}")
                    comb_seg = self.single_seg_run(im)
                    print(f"** END single_seg_run: {datetime.now()}")
                else:
                    print(f'FOV:{row["fovid"]} does not have nucleus or cellular color channels')
                    break
                
                if save_to_fms == True:
                    print("Uploading output file to FMS")
                    print(f"** START Write file: {datetime.now()}")

                    with TemporaryDirectory() as tmp_dir:
                        local_file_path = f'{tmp_dir}/{file_name}'
                        with ome_tiff_writer.OmeTiffWriter(local_file_path) as writer:
                            writer.save(comb_seg)
                        print(f"** START upload_combined_segmentation: {datetime.now()}")
                        self._uploader.upload_combined_segmentation(local_file_path, row["sourceimagefileid"])
                        print(f"** END upload_combined_segmentation: {datetime.now()}")

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



