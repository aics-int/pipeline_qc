import os

from dataclasses import dataclass
from pathlib import Path
from lkaccess import LabKey, QueryFilter
from ..configuration import AppConfig

@dataclass
class RunInfo:
    """
    Information about an algorithm run
    """
    run_id: int
    algorithm_id: int
    algorithm_name: str
    algorithm_version: str
    execution_date: str


class LabkeyProvider:
    """
    Interface for Labkey specific only operations
    """
    def __init__(self, labkey_client: LabKey):
        if labkey_client is None:
            raise AttributeError("labkey_client")
        self._labkey_client = labkey_client    
        self._ensure_netrc()

    def create_run_id(self, algorithm: str, algorithm_version: str, processing_date: str) -> int:
        """
        Create an algorithm "run" in Labkey and return the new run ID
        This is used to later link cell records with
        return: the run ID
        """        
        algorithm_id = self._labkey_client.select_first("processing",
                                                        "ContentGenerationAlgorithm",
                                                        filter_array=[
                                                            QueryFilter('Name', algorithm),
                                                            QueryFilter('Version', algorithm_version)
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

    def get_run_by_id(self, run_id: int) -> RunInfo:
        """
        Get algorithm run information for the given run ID
        return: 
        """
        ALGO_ID = "ContentGenerationAlgorithmId"
        ALGO_NAME = "ContentGenerationAlgorithmId/Name"
        ALGO_VERSION = "ContentGenerationAlgorithmId/Version"
        EXECUTION_DATE = "ExecutionDate"

        response = self._labkey_client.select_rows_as_list("processing", "run", 
                                   filter_array=[QueryFilter("RunId", run_id)], 
                                   columns=[ALGO_ID, ALGO_NAME, ALGO_VERSION, EXECUTION_DATE])

        if response is None or len(response) == 0:
            return None
        
        result = response[0]
        return RunInfo(run_id=run_id,
                       algorithm_id=int(result[ALGO_ID]),
                       algorithm_name=result[ALGO_NAME],
                       algorithm_version=result[ALGO_VERSION],
                       execution_date=result[EXECUTION_DATE])

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