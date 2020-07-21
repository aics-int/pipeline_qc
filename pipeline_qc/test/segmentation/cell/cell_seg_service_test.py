import pytest
import numpy
import os 

from unittest import mock
from unittest.mock import Mock
from pandas import Series, DataFrame
from pipeline_qc.segmentation.common.fov_file import FovFile
from pipeline_qc.segmentation.common.segmentation_result import SegmentationResult, ResultStatus

@pytest.mark.skipif(os.environ.get("USER", "") == "jenkins",
                    reason=f"Import errors on Jenkins. Can't install all necessary modules through gradle + setup.py.")
class TestCellSegmentationService:

    @pytest.fixture(autouse=True)
    def setup(self):
        from pipeline_qc.segmentation.cell.cell_seg_service import AppConfig, CellSegmentationService, CellSegmentationRepository
        self._mock_repository = Mock(spec=CellSegmentationRepository)
        self._cell_seg_service = CellSegmentationService(self._mock_repository, config=Mock(spec=AppConfig))

    def test_single_cell_segmentation_skips_existing_fov(self):
        from pipeline_qc.segmentation.cell.cell_seg_service import ResultStatus, SegmentationResult

        # Arrange       
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        self._mock_repository.segmentation_exists.return_value = True

        # Act
        result: SegmentationResult = self._cell_seg_service.single_cell_segmentation(fov, 
                                                                                         save_to_fms=False, 
                                                                                         save_to_filesystem=False, 
                                                                                         output_dir="", 
                                                                                         process_duplicates=False)

        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SKIPPED

    @pytest.mark.parametrize("image_data", 
                             [{"405nm": numpy.array([1, 2, 3]),"638nm": numpy.array([4, 5, 6])},
                              {"405nm": numpy.array([1]),"638nm": numpy.array([4, 5, 6]), "brightfield": numpy.array([7, 8, 9])}
                             ])
    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.file_processing_methods")
    def test_single_cell_segmentation_fails_on_incompatible_fov(self, mock_file_processing_methods: Mock, image_data):
        from pipeline_qc.segmentation.cell.cell_seg_service import ResultStatus, SegmentationResult
        
        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        self._mock_repository.segmentation_exists.return_value = False

        # Act
        result: SegmentationResult = self._cell_seg_service.single_cell_segmentation(fov, 
                                                                                         save_to_fms=False, 
                                                                                         save_to_filesystem=False, 
                                                                                         output_dir="", 
                                                                                         process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.FAILED
        assert result.message.startswith("Exception") == False


    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.file_processing_methods")
    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.SuperModel")
    def test_single_cell_segmentation_fails_on_empty_segmentation(self, mock_super_model: Mock, mock_file_processing_methods: Mock):
        from pipeline_qc.segmentation.cell.cell_seg_service import ResultStatus, SegmentationResult, SuperModel
        
        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        image_data = {
                      "405nm": numpy.array([1, 2, 3]),
                      "638nm": numpy.array([4, 5, 6]),
                      "brightfield": numpy.array([7, 8, 9])
                     }
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        mock_super_model.return_value.apply_on_single_zstack.return_value = None
        self._mock_repository.segmentation_exists.return_value = False

        # Act
        result: SegmentationResult = self._cell_seg_service.single_cell_segmentation(fov, 
                                                                save_to_fms=False, 
                                                                save_to_filesystem=False, 
                                                                output_dir="", 
                                                                process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.FAILED
        assert result.message.startswith("Exception") == False

    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.file_processing_methods")
    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.SuperModel")
    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.ome_tiff_writer.OmeTiffWriter")
    def test_single_cell_segmentation_happy_path_dual_camera(self, mock_tiff_writer: Mock, mock_super_model: Mock, mock_file_processing_methods: Mock):
        from pipeline_qc.segmentation.cell.cell_seg_service import ResultStatus, SegmentationResult, CellSegmentationService

        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        image_data = {
                      "405nm": numpy.array([1, 2, 3]),
                      "638nm": numpy.array([4, 5, 6]),
                      "brightfield": numpy.array([7, 8, 9])
                     }
        self._mock_repository.segmentation_exists.return_value = False    
        mock_file_processing_methods.split_image_into_channels.return_value = image_data        
        
        # Act
        result = self._cell_seg_service.single_cell_segmentation(fov, 
                                                                 save_to_fms=True, 
                                                                 save_to_filesystem=True, 
                                                                 output_dir="", 
                                                                 process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SUCCESS
        mock_super_model.assert_called_once_with("DNA_MEM_instance_LF_integration_two_camera")

    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.file_processing_methods")
    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.SuperModel")
    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.ome_tiff_writer.OmeTiffWriter")
    def test_single_cell_segmentation_happy_path_single_camera(self, mock_tiff_writer: Mock, mock_super_model: Mock, mock_file_processing_methods: Mock):
        from pipeline_qc.segmentation.cell.cell_seg_service import ResultStatus, SegmentationResult, CellSegmentationService

        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.1", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        image_data = {
                      "405nm": numpy.array([1, 2, 3]),
                      "638nm": numpy.array([4, 5, 6]),
                      "brightfield": numpy.array([7, 8, 9])
                     }
        self._mock_repository.segmentation_exists.return_value = False
        mock_file_processing_methods.split_image_into_channels.return_value = image_data        
        
        # Act
        result = self._cell_seg_service.single_cell_segmentation(fov, 
                                                                 save_to_fms=True, 
                                                                 save_to_filesystem=True, 
                                                                 output_dir="", 
                                                                 process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SUCCESS
        mock_super_model.assert_called_once_with("DNA_MEM_instance_LF_integration")

    @mock.patch("pipeline_qc.segmentation.cell.cell_seg_service.query_fovs")
    def test_get_fov_records(self, mock_query_fovs: Mock):
        # Arrange
        data = {
                "fovid": [63, 64, 65], 
                "workflow": [["Pipeline 4"], ["Pipeline 4.1"], ["Pipeline 4.2"]], 
                "localfilepath": ["/allen/file1.tiff", "/allen/file2.tiff", "/allen/file3.tiff"], 
                "sourceimagefileid": ["abc", "def", "1234"],
                "gene": ["LMNB1", "H2B", "AASV1"]
               }
        df = DataFrame(data=data)  
        mock_query_fovs.query_fovs.return_value = df

        # Act
        result = self._cell_seg_service.get_fov_records(workflows=None, plates=None, cell_lines=None, fovids=[63, 64, 65], only_from_fms=True)
        
        # Assert
        assert result is not None
        assert len(result) == 3
        assert result[0].fov_id == 63
        assert result[1].fov_id == 64
        assert result[2].fov_id == 65
