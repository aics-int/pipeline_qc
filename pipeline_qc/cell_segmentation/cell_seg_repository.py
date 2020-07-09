import os
import logging

from pathlib import Path
from lkaccess import LabKey, QueryFilter
from datetime import datetime
from aicsfiles import FileManagementSystem
from aicsfiles.filter import Filter
from .configuration import AppConfig

# Algorithm information
# must match existing ContentGenerationAlgorithm name and version in Labkey
ALGORITHM = "dna_cell_segmentation_ML_v1" 
ALGORITHM_VERSION = "0.1.0"

class ContentTypes(object):
    """
    Labkey ContentTypes (processing.ContentType table)
    """
    NucSeg = "Nucleus segmentation"
    MembSeg = "Membrane segmentation"
    NucContour = "Nucleus contour"
    MembContour = "Membrane contour"

class CellSegmentationRepository:
    """
    Interface for persistence (FMS/Labkey) operations on segmentation files
    """
    def __init__(self, fms_client: FileManagementSystem, labkey_client: LabKey, config: AppConfig):
        if fms_client is None:
            raise AttributeError("fms_client")
        if labkey_client is None:
            raise AttributeError("labkey_client")
        if config is None:
            raise AttributeError("config")
        self._fms_client = fms_client
        self._labkey_client = labkey_client
        self._config = config        
        self._ensure_netrc()

    def upload_combined_segmentation(self, combined_segmentation_path: str, input_file_id: str):
        """
        Augment with proper metadata and upload a combined segmentation files to FMS
        :param: combined_segmentation_path: combined segmentation output file path
        :param: input_file_id: file ID of the input image used to produce this segmentation (used to gather metadata)
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
        #         },
        #         "1": {
        #           ...
        #         },
        #         ...
        #     }
        # }

        # Channel 0 = Nucleus segmentation
        # Channel 1 = Membrane segmentation
        # Channel 2 = Nucleus contour
        # Channel 3 = Membrane contour

        processing_date: str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        run_id = self._create_run_id(processing_date)

        # Initialize metadata from input file's metadata, or start over if not available
        metadata = self._get_file_metadata(input_file_id) or {}
        metadata.update({"file": {"file_type": "image"}})

        metadata["content_processing"] = {
            "channels": {
                "0": self._channel_metadata_block(ContentTypes.NucSeg, processing_date, run_id),
                "1": self._channel_metadata_block(ContentTypes.MembSeg, processing_date, run_id),
                "2": self._channel_metadata_block(ContentTypes.NucContour, processing_date, run_id),
                "3": self._channel_metadata_block(ContentTypes.MembContour, processing_date, run_id)
            }
        }

        self._fms_client.upload_file_sync(combined_segmentation_path, metadata, timeout=self._config.fms_timeout_in_seconds)

    def segmentation_exists(self, filename: str):
        """
        Check whether the given segmentation file has already been persisted 
        param: filename: segmentation file name
        return: True if file already exists, False otherwise
        """
        query = Filter().with_file_name(filename)
        result = self._fms_client.query_files(query)
        return (result is not None and len(result) > 0)

    def _channel_metadata_block(self, content_type: str, processing_date: str, run_id: int):
        """
        Build and return a metadata block for a given channel
        param: content_type: content type to identify the channel's contents
        param: processing_date: content processing date
        """
        return {
                "algorithm": ALGORITHM,
                "algorithm_version": ALGORITHM_VERSION,
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

    def _create_run_id(self, processing_date: str) -> int:
        """
        Create an algorithm "run" in Labkey and return the new run ID
        This is used to later link cell records with
        return: the run ID
        """        
        algorithm_id = self._labkey_client.select_first("processing",
                                                        "ContentGenerationAlgorithm",
                                                        filter_array=[
                                                            QueryFilter('Name', ALGORITHM),
                                                            QueryFilter('Version', ALGORITHM_VERSION)
                                                        ])["ContentGenerationAlgorithmId"]

        row = {
            'ContentGenerationAlgorithmId': algorithm_id,
            'ExecutionDate': processing_date,
            'Notes': None
        }
        response = self._labkey_client.insert_rows("processing", "Run", rows=[row])

        if response is None or "rows" not in response or len(response["rows"]) == 0:
            raise Exception(f"Failed to create Run ID or unable to retrieve result from Labkey.")

        return int(response['rows'][0]['runid'])
        

    def _ensure_netrc(self):
        """
        Ensure presence of Labkey .netrc credentials file
        This file is required for authentication necessary for Update / Insert operations in Labkey
        """
        # This file must exist for uploads to proceed
        netrc = Path.home() / ('_netrc' if os.name == 'nt' else '.netrc')
        if not netrc.exists():
            raise Exception(f"{netrc} was not found. It must exist with appropriate credentials for "
                            f"uploading data to labkey."
                            f"See https://www.labkey.org/Documentation/wiki-page.view?name=netrc for setup instructions.")