import pytest

from unittest.mock import Mock, call
from pipeline_qc.segmentation.common.labkey_provider import LabkeyProvider, LabKey

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