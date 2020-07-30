import pytest
import numpy
import os 

from unittest import mock
from unittest.mock import Mock
from pandas import Series, DataFrame
from pipeline_qc.segmentation.common.fov_file import FovFile
from pipeline_qc.segmentation.common.segmentation_result import SegmentationResult, ResultStatus
from pipeline_qc.segmentation.structure.structures import StructureInfo

@pytest.mark.skipif(os.environ.get("USER", "") == "jenkins",
                    reason=f"Import errors on Jenkins. Can't install all necessary modules through gradle + setup.py.")
class TestStructureSegmentationService:

    @pytest.fixture(autouse=True)
    def setup(self):
        from pipeline_qc.segmentation.structure.structure_seg_service import AppConfig, StructureSegmentationService, StructureSegmentationRepository
        self._mock_repository = Mock(spec=StructureSegmentationRepository)
        self._structure_seg_service = StructureSegmentationService(self._mock_repository, config=Mock(spec=AppConfig))

    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.query_fovs")
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
        result = self._structure_seg_service.get_fov_records(workflows=None, plates=None, cell_lines=None, fovids=[63, 64, 65], only_from_fms=True)
        
        # Assert
        assert result is not None
        assert len(result) == 3
        assert result[0].fov_id == 63
        assert result[1].fov_id == 64
        assert result[2].fov_id == 65    

    def test_structure_segmentation_skips_existing_fov(self):
        # Arrange       
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        self._mock_repository.segmentation_exists.return_value = True

        # Act
        result: SegmentationResult = self._structure_seg_service.structure_segmentation(fov, 
                                                                                        save_to_fms=False, 
                                                                                        save_to_filesystem=False, 
                                                                                        output_dir="", 
                                                                                        process_duplicates=False)

        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SKIPPED

    def test_structure_segmentation_fails_on_unsupported_structure(self):
        # Arrange       
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", 
                      source_image_file_id="abcdef123456", gene="NOT_A_STRUCTURE")
        self._mock_repository.segmentation_exists.return_value = False

        # Act     
        result: SegmentationResult = self._structure_seg_service.structure_segmentation(fov, 
                                                                                        save_to_fms=False, 
                                                                                        save_to_filesystem=False, 
                                                                                        output_dir="", 
                                                                                        process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.FAILED
                                                                                           
    @pytest.mark.parametrize("image_data,ml", 
                             [({"488nm": numpy.array([1, 2, 3])}, True),
                              ({"488nm": numpy.array([1]),"638nm": numpy.array([4, 5, 6])}, True),
                              ({"405nm": numpy.array([1, 2, 3])}, False),
                              ({"488nm": numpy.array([1])}, False)
                             ])
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.Structures")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.file_processing_methods")
    def test_structure_segmentation_fails_on_incompatible_fov(self, mock_file_processing_methods: Mock, mock_structures: Mock, image_data, ml):        
        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        mock_structures.get.return_value = StructureInfo(gene="LMNB1", ml=ml, ml_model="test", algorithm_name="test", algorithm_version="1.0")
        self._mock_repository.segmentation_exists.return_value = False


        # Act     
        result: SegmentationResult = self._structure_seg_service.structure_segmentation(fov, 
                                                                                        save_to_fms=False, 
                                                                                        save_to_filesystem=False, 
                                                                                        output_dir="", 
                                                                                        process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.FAILED
        assert result.message.startswith("Exception") == False

    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.Structures")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.file_processing_methods")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.SuperModel")
    def test_structure_segmentation_ml_fails_on_empty_segmentation(self, mock_super_model: Mock, mock_file_processing_methods: Mock, mock_structures: Mock):       
        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        image_data = {
                      "488nm": numpy.array([1, 2, 3]),
                      "638nm": numpy.array([4, 5, 6])
                     }
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        mock_super_model.return_value.apply_on_single_zstack.return_value = None
        mock_structures.get.return_value = StructureInfo(gene="LMNB1", ml=True, ml_model="test", algorithm_name="test", algorithm_version="1.0")
        self._mock_repository.segmentation_exists.return_value = False

        # Act     
        result: SegmentationResult = self._structure_seg_service.structure_segmentation(fov, 
                                                                                        save_to_fms=False, 
                                                                                        save_to_filesystem=False, 
                                                                                        output_dir="", 
                                                                                        process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.FAILED
        assert result.message.startswith("Exception") == False

    def test_structure_segmentation_legacy_fails_on_empty_segmentation(self):
        pass #TODO

    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.Structures")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.file_processing_methods")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.SuperModel")
    def test_structure_segmentation_ml_happy_path(self, mock_super_model: Mock, mock_file_processing_methods: Mock, mock_structures: Mock):       
        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="H2B")
        image_data = {
                      "488nm": numpy.array([1, 2, 3]),
                      "638nm": numpy.array([4, 5, 6])
                     }
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name="ML H2B Structure Segmentation", algorithm_version="0.1.0")
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        mock_structures.get.return_value = StructureInfo(gene="LMNB1", ml=True, ml_model="test", algorithm_name="test", algorithm_version="1.0")
        self._mock_repository.segmentation_exists.return_value = False

        # Act     
        result: SegmentationResult = self._structure_seg_service.structure_segmentation(fov, 
                                                                                        save_to_fms=False, 
                                                                                        save_to_filesystem=False, 
                                                                                        output_dir="", 
                                                                                        process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SUCCESS
        mock_super_model.assert_called_once_with(struct_info.ml_model)

    def test_structure_segmentation_ml_happy_path(self):
        pass #TODO