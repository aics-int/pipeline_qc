import os
import logging

from pathlib import Path
from lkaccess import LabKey, QueryFilter
from datetime import datetime
from aicsfiles import FileManagementSystem
from aicsfiles.filter import Filter
from ..configuration import AppConfig
from ..common.labkey_provider import LabkeyProvider
from .structures import StructureInfo

class ContentTypes(object):
    """
    Labkey ContentTypes (processing.ContentType table)
    """
    StructureSeg = "Structure segmentation"
    StructureContour = "Structure contour"

class StructureSegmentationRepository:
    """
    Interface for persistence (FMS/Labkey) operations on segmentation files
    """
    def __init__(self, fms_client: FileManagementSystem, labkey_provider: LabkeyProvider, config: AppConfig):
        if fms_client is None:
            raise AttributeError("fms_client")
        if labkey_provider is None:
            raise AttributeError("labkey_provider")
        if config is None:
            raise AttributeError("config")
        self._fms_client = fms_client
        self._labkey_provider = labkey_provider
        self._config = config        

    def upload_structure_segmentation(self, 
                                      structure_info: StructureInfo, 
                                      source_file_id: str,
                                      segmentation_path: str, 
                                      contour_path: str = None):
        """
        Augment with proper metadata and upload a structure segmentation file to FMS
        :param: structure_info: information about the segmented structure
        :param: structure_segmentation_path: structure segmentation output file path
        :param: source_file_id: file ID of the input image used to produce this segmentation (used to gather metadata)
        """

        # Minimal Metadata structure:
        #
        # "file": {
        #     "file_type": "image"
        # },
        # "content_processing": {
        #     "channels": {
        #         "0": {
        #             "algorithm": <algorithm name, as recorded in Labkey>,
        #             "algorithm_version": <algorithm version, as recorded in Labkey>,
        #             "content_type": <ContentType, as recorded in Labkey>,
        #             "processing_date": <processing date>,
        #             "run_id": <run id>
        #         }
        #     }
        # }

        # Channel 0 = Structure segmentation


        processing_date: str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        run_id = self._labkey_provider.create_run_id(structure_info.algorithm_name, structure_info.algorithm_version, processing_date)

        # Initialize metadata from input file's metadata, or start over if not available
        metadata = self._get_file_metadata(source_file_id) or {}
        metadata.update({"file": {"file_type": "image"}})

        # Contour
        if contour_path is not None:
            metadata["content_processing"] = {
                "channels": {
                    "0": self._channel_metadata_block(ContentTypes.StructureContour, 
                                                      structure_info.algorithm_name, 
                                                      structure_info.algorithm_version, 
                                                      processing_date, 
                                                      run_id)
                }
            }
            self._fms_client.upload_file_sync(contour_path, metadata, timeout=self._config.fms_timeout_in_seconds)

        # Segmentation
        metadata["content_processing"] = {
            "channels": {
                "0": self._channel_metadata_block(ContentTypes.StructureSeg, 
                                                  structure_info.algorithm_name, 
                                                  structure_info.algorithm_version, 
                                                  processing_date, 
                                                  run_id)
            }
        }

        self._fms_client.upload_file_sync(segmentation_path, metadata, timeout=self._config.fms_timeout_in_seconds)


    def segmentation_exists(self, filename: str, fov_id: int):
        """
        Check whether the given FOV has already been segmented
        param: fov_id: the FOV id
        return: True if segmentation already exists, False otherwise
        """
        # 1) check that file exists
        query = Filter().with_file_name(filename)
        result = self._fms_client.query_files(query)
        file_exists = (result is not None and len(result) > 0) 
        if not file_exists:
            return False  

        # 2) check that run_id exists with current algorithm
        pass #TODO

    def _channel_metadata_block(self, content_type: str, algorithm: str, algorithm_version: str, processing_date: str, run_id: int):
        """
        Build and return a metadata block for a given channel
        param: content_type: content type to identify the channel's contents
        param: processing_date: content processing date
        """
        return {
                "algorithm": algorithm,
                "algorithm_version": algorithm_version,
                "content_type": content_type,
                "processing_date": processing_date,
                "run_id": run_id
                }

    def _get_file_metadata(self, file_id: str):
        """
        Get file metadata from FMS
        param: file_id: ID of the file in FMS
        return: file metadata if the file was found, None otherwise
        """
        query = Filter().with_file_id(file_id)
        result = self._fms_client.query_files(query)

        return result[0] if result is not None and len(result) > 0 else None
