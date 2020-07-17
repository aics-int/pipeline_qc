import os
import logging

from pathlib import Path
from lkaccess import LabKey, QueryFilter
from datetime import datetime
from aicsfiles import FileManagementSystem
from aicsfiles.filter import Filter
from ..configuration import AppConfig
from ..common.labkey_provider import LabkeyProvider


class ContentTypes(object):
    """
    Labkey ContentTypes (processing.ContentType table)
    """
    StructureSeg = "Structure segmentation"
    StructureContour = "Structure contour"

class CellSegmentationRepository:
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

    def upload_structure_segmentation(self): #TODO
        pass

    def segmentation_exists(self, fov_id: int):
        """
        Check whether the given FOV has already been segmented
        param: fov_id: the FOV id
        return: True if file already exists, False otherwise
        """
        pass #TODO

    def _channel_metadata_block(self, algorithm: str, algorithm_version: str, content_type: str, processing_date: str, run_id: int):
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
