import numpy as np
import os
import aicsimageio
import traceback
import logging

from typing import List
from enum import Enum
from pandas import Series
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
from aicsimageio.writers import ome_tiff_writer
from pipeline_qc.image_qc_methods import file_processing_methods, query_fovs
from model_zoo_3d_segmentation.zoo import SuperModel
from .structure_seg_repository import StructureSegmentationRepository
from ..configuration import AppConfig
from ..common.fov_file import FovFile
from ..common.segmentation_result import SegmentationResult, ResultStatus

    

class StructureSegmentationService:
    """
    Structure Segmentation Service
    Exposes functionality to perform specific structure segmentations on FOVs
    """


    def __init__(self, repository: StructureSegmentationRepository, config: AppConfig):
        if repository is None:
            raise AttributeError("repository")
        if config is None:
            raise AttributeError("app_config")
        self._repository = repository
        self._config = config
        self.log = logging.getLogger(__name__)

    def get_fov_records(self, workflows: List, plates: List, cell_lines: List, fovids: List, only_from_fms:bool) -> List[FovFile]:
        """
        Query for FOV records and return results in list form
        """
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                         only_from_fms=only_from_fms, labkey_host=self._config.labkey_host, labkey_port=self._config.labkey_port)
        
        fovs = []
        for index, row in query_df.iterrows():
            fovs.append(FovFile.from_dataframe_row(row))

        return fovs

    def structure_segmentation(self, 
                               fov: FovFile, 
                               save_to_fms: bool, 
                               save_to_filesystem: bool, 
                               output_dir: str, 
                               process_duplicates: bool) -> SegmentationResult: 
        """
        Run segmentation process for a single FOV
        :param: fov: FOV record
        :param: save_to_fms: indicate whether to save segmentation output to FMS
        :param: save_to_filesystem: indicate whether to save segmentation output to output_dir
        :param: output_dir: output directory path when saving to file system (can be network / isilon path)
        :param: process_duplicates: indicate whether to process or skip fov if segmentation already exists in FMS
        """               

        fov_id = fov.fov_id
        local_file_path = fov.local_file_path
        source_file_id = fov.source_image_file_id
        
        try:
            if not process_duplicates and self._repository.segmentation_exists(fov_id):
                msg = f"FOV {fov_id} has already been segmented"
                self.log.info(msg)
                return SegmentationResult(fov_id=fov_id, status=ResultStatus.SKIPPED, message=msg)

            #TODO create segmentable image based on structure
            #TODO segment
            
            if save_to_fms:
                self.log.info("Uploading output file to FMS")
                #TODO

            if save_to_filesystem:
                self.log.info("Saving output file to filesystem")
                #TODO

            return SegmentationResult(fov_id=fov_id, status=ResultStatus.SUCCESS)
        except Exception as ex:
            msg = f"Exception while processing FOV {fov_id}: {str(ex)}\n{traceback.format_exc()}"
            self.log.info(msg)
            return SegmentationResult(fov_id=fov_id, status=ResultStatus.FAILED, message=msg)

    

    def _create_segmentable_image_ml(self, fov: FovFile):
        """
        Create a segmentable image by picking the right channels
        ML structure segmentation requires an image with 2 channels: 488nm (structure) and 638nm (membrane)
        """
        aicsimageio.use_dask(False) # disable dask image reads to avoid losing performance when running on GPU nodes

        channel_dict = file_processing_methods.split_image_into_channels(fov.local_file_path, fov.source_image_file_id)

        full_im_list = list()

        for channel in ['488nm', '638nm']:
            for channel_name, channel_array in channel_dict.items():
                if channel_array.shape[0] <= 1: 
                    return None # not a 3D image
                if channel_name == channel:
                    full_im_list.append(channel_array)

        return np.array(full_im_list)

    def _create_segmentable_image_legacy(self, fov: FovFile):
        """
        Create a segmentable image by picking the right channels
        Legacy structure segmentation requires the image structure channel only
        """
        aicsimageio.use_dask(False) # disable dask image reads to avoid losing performance when running on GPU nodes

        channel_dict = file_processing_methods.split_image_into_channels(fov.local_file_path, fov.source_image_file_id)

        struct_channel = channel_dict["488nm"]
        if struct_channel.shape[0] <= 1:
            return None # not a 3D image
        
        return struct_channel
