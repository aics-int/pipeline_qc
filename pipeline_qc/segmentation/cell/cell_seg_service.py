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
from .cell_seg_repository import CellSegmentationRepository
from ..configuration import AppConfig
from ..common.fov_file import FovFile

class ResultStatus(Enum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass
class CellSegmentationResult:
    fov_id: int
    status: ResultStatus
    message: str = ""

    def __str__(self):
        return f"FOV {self.fov_id}: {self.status.value}\n{self.message}"
    

class CellSegmentationService:
    """
    Single cell ML Segmentation Service
    Exposes functionality to perform single cell segmentations of FOVs using the latest ML segmentation algorithms
    This service wraps the core ML Segmentation code from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse
    """

    # Constants
    MODEL_SINGLE_CAMERA = "DNA_MEM_instance_LF_integration"
    MODEL_DUAL_CAMERA = "DNA_MEM_instance_LF_integration_two_camera"

    def __init__(self, repository: CellSegmentationRepository, config: AppConfig):
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

    def single_cell_segmentation(self, 
                                 fov: FovFile, 
                                 save_to_fms: bool, 
                                 save_to_filesystem: bool, 
                                 output_dir: str, 
                                 process_duplicates: bool) -> CellSegmentationResult: 
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
        model = self.MODEL_SINGLE_CAMERA if fov.is_single_camera else self.MODEL_DUAL_CAMERA
        
        try:
            file_name = self._get_seg_filename(local_file_path)

            if not process_duplicates and self._repository.segmentation_exists(file_name):
                msg = f"FOV {fov_id} has already been segmented"
                self.log.info(msg)
                return CellSegmentationResult(fov_id=fov_id, status=ResultStatus.SKIPPED, message=msg)
            
            im = self._create_segmentable_image(fov)
            if im.shape[0] != 3:
                msg = f"FOV {fov_id} does not have nucleus or cellular color channels"
                self.log.info(msg)
                return CellSegmentationResult(fov_id=fov_id, status=ResultStatus.SKIPPED, message=msg)
            
            self.log.info(f'Running Segmentation on FOV {fov_id}')

            combined_segmentation = self._segment_from_model(im, model)
            if combined_segmentation is None:
                msg = f"FOV {fov_id} could not be segmented: returned empty result"
                self.log.info(msg)
                return CellSegmentationResult(fov_id=fov_id, status=ResultStatus.FAILED, msg=msg)

            if save_to_fms:
                self.log.info("Uploading output file to FMS")

                with TemporaryDirectory() as tmp_dir:
                    local_file_path = f'{tmp_dir}/{file_name}'
                    with ome_tiff_writer.OmeTiffWriter(local_file_path) as writer:
                        writer.save(combined_segmentation)
                    self._repository.upload_combined_segmentation(local_file_path, source_file_id)

            if save_to_filesystem:
                self.log.info("Saving output file to filesystem")
                with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{file_name}') as writer:
                    writer.save(combined_segmentation)

            return CellSegmentationResult(fov_id=fov_id, status=ResultStatus.SUCCESS)

        except Exception as ex:
            msg = f"Exception while processing FOV {fov_id}: {str(ex)}\n{traceback.format_exc()}"
            self.log.info(msg)
            return CellSegmentationResult(fov_id=fov_id, status=ResultStatus.FAILED, message=msg)

    def _segment_from_model(self, image, model):
        sm = SuperModel(model)

        return sm.apply_on_single_zstack(input_img=image)

    def _create_segmentable_image(self, fov: FovFile):
        """
        Create a segmentable image by picking the right channels
        Nucleus/Membrane segmentation requires an image with 3 channels: 405nm, 638nm and brightfield
        """
        aicsimageio.use_dask(False) # disable dask image reads to avoid losing performance when running on GPU nodes

        channel_dict = file_processing_methods.split_image_into_channels(fov.local_file_path, fov.source_image_file_id)

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