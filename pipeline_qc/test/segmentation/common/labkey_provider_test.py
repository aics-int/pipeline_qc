import pytest

from unittest.mock import Mock, call
from pipeline_qc.segmentation.common.labkey_provider import LabkeyProvider, LabKey, RunInfo

class TestCellSegmentationRepository:

    @pytest.fixture(autouse=True)
    def setup(self):
        self._mock_labkey_client = Mock(spec=LabKey)
        self._labkey_provider = LabkeyProvider(self._mock_labkey_client)

    def test_create_run_id(self):
        # Arrange
        self._mock_labkey_client.insert_rows.return_value = {"rows": [{"runid": 1234}]}
        self._mock_labkey_client.select_first.return_value = {"ContentGenerationAlgorithmId": 999}

        # Act
        run_id = self._labkey_provider.create_run_id("segmentation", "2.0", "01/01/2021")

        # Assert
        assert run_id == 1234

    def test_get_run_by_id_found(self):
        # Arrange
        run_id = 1234
        expected_algo_id = 999
        expected_algo_name = "test_algorithm"
        expected_algo_version = "1.0"
        expected_execution_date = "2020-07-24 19:46:12.000"

        self._mock_labkey_client.select_rows_as_list.return_value = [{
            'ContentGenerationAlgorithmId': expected_algo_id, 
            'ContentGenerationAlgorithmId/Name': expected_algo_name,
            'ContentGenerationAlgorithmId/Version': expected_algo_version,  
            'ExecutionDate': expected_execution_date
        }]

        # Act
        run_info: RunInfo = self._labkey_provider.get_run_by_id(run_id)

        # Assert
        assert run_info.run_id == run_id
        assert run_info.algorithm_id == expected_algo_id
        assert run_info.algorithm_name == expected_algo_name
        assert run_info.algorithm_version == expected_algo_version
        assert run_info.execution_date == expected_execution_date

    
    def test_get_run_by_id_not_found(self):
        # Arrange
        self._mock_labkey_client.select_rows_as_list.return_value = []

        # Assert
        assert self._labkey_provider.get_run_by_id(1234) == None