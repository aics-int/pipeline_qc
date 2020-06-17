import numpy as np
import os
import aicsimageio

from pandas import Series
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
from aicsimageio.writers import ome_tiff_writer
from pipeline_qc.image_qc_methods import file_processing_methods
from pipeline_qc.image_qc_methods.cell_seg_repository import CellSegmentationRepository
from model_zoo_3d_segmentation.zoo import SuperModel
from .configuration import AppConfig

# Constants
MODEL = "DNA_MEM_instance_LF_integration_two_camera"

class CellSegmentationService:
    """
    Single cell ML Segmentation Service
    Exposes functionality to perform single cell segmentations of FOVs using the latest ML segmentation algorithms
    This service wraps the core ML Segmentation code from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse
    """

    def __init__(self, repository: CellSegmentationRepository, config: AppConfig):
        if repository is None:
            raise AttributeError("repository")
        if config is None:
            raise AttributeError("app_config")
        self._repository = repository
        self._config = config

    def single_cell_segmentation(self, row: Series, save_to_fms: bool, save_to_filesystem: bool, output_dir: str, process_duplicates: bool): 
        """
        Run segmentation process for a single FOV
        :param: row: FOV information as a pandas Dataframe row
        :param: save_to_fms: indicate whether to save segmentation output to FMS
        :param: save_to_filesystem: indicate whether to save segmentation output to output_dir
        :param: output_dir: output directory path when saving to file system (can be network / isilon path)
        :param: process_duplicates: indicate whether to process or skip fov if segmentation already exists in FMS
        """                                

        file_name = self._get_seg_filename(row['localfilepath'])

        if not process_duplicates and self._repository.segmentation_exists(file_name):
            print(f'FOV:{row["fovid"]} has already been segmented') 
            return
        
        print(f'Running Segmentation on fov:{row["fovid"]}')
        im = self._create_segmentable_image(row['localfilepath'], row['sourceimagefileid'])
        if im.shape[0] ==3:
            comb_seg = self._segment_from_model(im, MODEL)
        else:
            print(f'FOV:{row["fovid"]} does not have nucleus or cellular color channels')
            return
        
        if save_to_fms == True:
            print("Uploading output file to FMS")

            with TemporaryDirectory() as tmp_dir:
                local_file_path = f'{tmp_dir}/{file_name}'
                with ome_tiff_writer.OmeTiffWriter(local_file_path) as writer:
                    writer.save(comb_seg)
                self._repository.upload_combined_segmentation(local_file_path, row["sourceimagefileid"])

        if save_to_filesystem == True:
            print("Saving output file to Isilon")
            with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{file_name}') as writer:
                writer.save(comb_seg)

        return

    def _segment_from_model(self, image, model):
        sm = SuperModel(model)

        return sm.apply_on_single_zstack(input_img=image, inputCh=[0, 1, 2])

    def _create_segmentable_image(self, localfilepath, sourceimagefileid):
        aicsimageio.use_dask(False) # disable dask image reads to avoid losing performance when running on GPU nodes
        channel_dict = file_processing_methods.split_image_into_channels(localfilepath, sourceimagefileid)

        full_im_list = list()
        for channel in ['405nm', '638nm', 'brightfield']:
            for key, value in channel_dict.items():
                if key == channel:
                    full_im_list.append(value)

        return np.array(full_im_list)

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