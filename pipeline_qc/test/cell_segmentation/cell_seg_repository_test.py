import pytest

from unittest.mock import Mock, call
from aicsfiles import FileManagementSystem
from pipeline_qc.cell_segmentation.cell_seg_repository import CellSegmentationRepository, FileManagementSystem
from pipeline_qc.cell_segmentation.configuration import AppConfig

class TestCellSegmentationRepository:

    @pytest.fixture(autouse=True)
    def setup(self):
        self._mock_fms_client = Mock(spec=FileManagementSystem)
        self._cell_seg_repository = CellSegmentationRepository(fms_client=self._mock_fms_client, config=Mock(spec=AppConfig))

    def test_upload_combined_segmentation_no_initial_metadata(self):
        # Arrange
        input_file_path = "test/input.tiff"
        combined_seg_path = "test/combined_seg.tiff"
        self._mock_fms_client.query_files.return_value = None

        # Act
        self._cell_seg_repository.upload_combined_segmentation(combined_seg_path, input_file_path)

        # Assert
        self._mock_fms_client.upload_file.assert_called_once()
        upload_path = self._mock_fms_client.upload_file.call_args[0][0]
        assert upload_path == combined_seg_path
        metadata = self._mock_fms_client.upload_file.call_args[0][1]
        assert metadata["file"]["file_type"] == "image"
        assert len(metadata["content_processing"]["channels"]) == 4  # 4 channels
        assert metadata["content_processing"]["channels"]["0"]["content_type"] == "Nucleus segmentation"
        assert metadata["content_processing"]["channels"]["1"]["content_type"] == "Membrane segmentation"
        assert metadata["content_processing"]["channels"]["2"]["content_type"] == "Nucleus contour"
        assert metadata["content_processing"]["channels"]["3"]["content_type"] == "Membrane contour"

    def test_upload_combined_segmentation_with_initial_metadata(self):
        # Arrange
        input_file_path = "test/input.tiff"
        combined_seg_path = "test/combined_seg.tiff"
        self._mock_fms_client.query_files.return_value = [{"microscopy": {"fov_id": "9999"}}]

        # Act
        self._cell_seg_repository.upload_combined_segmentation(combined_seg_path, input_file_path)

        # Assert
        metadata = self._mock_fms_client.upload_file.call_args[0][1]
        assert metadata["file"]["file_type"] == "image"
        assert len(metadata["content_processing"]["channels"]) == 4  # 4 channels
        assert metadata["microscopy"]["fov_id"] == "9999"

    def test_segmentation_exists_found(self):
         # Arrange
        self._mock_fms_client.query_files.return_value = [{"file_id": "1234", "file_name": "segmentation.ome.tiff"}]      
        
        # Act
        exists = self._cell_seg_repository.segmentation_exists("segmentation.ome.tiff")

        # Assert
        assert exists == True

    @pytest.mark.parametrize("result", [None, []])
    def test_segmentation_exists_not_found(self, result):
         # Arrange
        self._mock_fms_client.query_files.return_value = result      
        
        # Act
        exists = self._cell_seg_repository.segmentation_exists("segmentation.ome.tiff")

        # Assert
        assert exists == False
