import pytest
import numpy

from unittest import mock
from unittest.mock import Mock
from pandas import Series
from pipeline_qc.cell_segmentation.cell_seg_service import SuperModel, ResultStatus, AppConfig, CellSegmentationService, CellSegmentationResult, CellSegmentationRepository

class TestCellSegmentationService:
    row = Series(index=["fovid", "localfilepath", "sourceimagefileid"],
                 data=[63, "/allen/aics/some/place/file.tiff", "abcdef123456"])

    @pytest.fixture(autouse=True)
    def setup(self):
        self._mock_repository = Mock(spec=CellSegmentationRepository)
        self._cell_seg_service = CellSegmentationService(self._mock_repository, config=Mock(spec=AppConfig))

    def test_single_cell_segmentation_skips_existing_fov(self):
        # Arrange       
        self._mock_repository.segmentation_exists.return_value = True

        # Act
        result: CellSegmentationResult = self._cell_seg_service.single_cell_segmentation(self.row, 
                                                                                         save_to_fms=False, 
                                                                                         save_to_filesystem=False, 
                                                                                         output_dir="", 
                                                                                         process_duplicates=False)

        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SKIPPED

    @mock.patch("pipeline_qc.cell_segmentation.cell_seg_service.file_processing_methods")
    def test_single_cell_segmentation_skips_incompatible_fov(self, mock_file_processing_methods):
        # Arrange
        image_data = {
                      "405nm": [1, 2, 3],
                      "638nm": [4, 5, 6]
                     }
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        self._mock_repository.segmentation_exists.return_value = False

        # Act
        result: CellSegmentationResult = self._cell_seg_service.single_cell_segmentation(self.row, 
                                                                                         save_to_fms=False, 
                                                                                         save_to_filesystem=False, 
                                                                                         output_dir="", 
                                                                                         process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SKIPPED

    @mock.patch("pipeline_qc.cell_segmentation.cell_seg_service.ome_tiff_writer.OmeTiffWriter")
    @mock.patch("pipeline_qc.cell_segmentation.cell_seg_service.SuperModel")
    def test_single_cell_segmentation_happy_path(self, mock_tiff_writer, mock_super_model):
        # Arrange
        image_data = {
                      "405nm": [1, 2, 3],
                      "638nm": [4, 5, 6],
                      "brightfield": [7, 8, 9]
                     }
        self._mock_repository.segmentation_exists.return_value = False
        mock_file_processing_methods = Mock()      
        mock_file_processing_methods.split_image_into_channels.return_value = image_data        
        
        # Act
        result: CellSegmentationResult = None
        with mock.patch("pipeline_qc.cell_segmentation.cell_seg_service.file_processing_methods", mock_file_processing_methods):
            result = self._cell_seg_service.single_cell_segmentation(self.row, 
                                                                    save_to_fms=True, 
                                                                    save_to_filesystem=True, 
                                                                    output_dir="", 
                                                                    process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SUCCESS