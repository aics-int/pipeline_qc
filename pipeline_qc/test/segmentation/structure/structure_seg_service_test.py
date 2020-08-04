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
        from pipeline_qc.segmentation.structure.structure_seg_service import AppConfig, StructureSegmentationService, StructureSegmentationRepository, StructureSegmenter
        self._mock_legacy_segmenter = Mock(spec=StructureSegmenter)
        self._mock_repository = Mock(spec=StructureSegmentationRepository)
        self._structure_seg_service = StructureSegmentationService(self._mock_legacy_segmenter, self._mock_repository, config=Mock(spec=AppConfig))

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
                             [({"488nm": numpy.array([1, 2, 3])}, True), # missing channel
                              ({"488nm": numpy.array([1]), "638nm": numpy.array([4, 5, 6])}, True), # missing dimentions
                              ({"488nm": numpy.array([1, 2, 3]), "561nm": numpy.array([4, 5, 6]), "638nm": numpy.array([7, 8, 9])}, True), # dual structure channel
                              ({"405nm": numpy.array([1, 2, 3])}, False), # missing channel
                              ({"488nm": numpy.array([1])}, False), # missing dimentions
                              ({"488nm": numpy.array([1, 2, 3]), "561nm": numpy.array([4, 5, 6])}, False) # dual structure channel
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
        assert "incompatible" in result.message

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

    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.Structures")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.file_processing_methods")
    def test_structure_segmentation_legacy_fails_on_empty_segmentation(self, mock_file_processing_methods: Mock, mock_structures: Mock):
        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        image_data = {
                      "488nm": numpy.array([1, 2, 3]),
                      "638nm": numpy.array([4, 5, 6])
                     }
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        mock_structures.get.return_value = StructureInfo(gene="LMNB1", ml=False, ml_model=None, algorithm_name="test", algorithm_version="1.0")
        self._mock_repository.segmentation_exists.return_value = False
        self._mock_legacy_segmenter.process_img.return_value = None

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

    @pytest.mark.parametrize("image_data,struct_channel", 
                             [({"488nm": numpy.array([1, 2, 3]), "638nm": numpy.array([4, 5, 6])}, "488nm"),
                              ({"561nm": numpy.array([1, 2, 3]), "638nm": numpy.array([4, 5, 6])}, "561nm")
                             ])
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.Structures")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.file_processing_methods")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.SuperModel")
    def test_structure_segmentation_ml_happy_path(self, mock_super_model: Mock, mock_file_processing_methods: Mock, mock_structures: Mock, image_data, struct_channel):       
        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="H2B")
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name="ML H2B Structure Segmentation", algorithm_version="0.1.0")
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        mock_structures.get.return_value = struct_info
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
        mock_super_model.return_value.apply_on_single_zstack.assert_called_once()
        image_array = mock_super_model.return_value.apply_on_single_zstack.call_args[1]["input_img"]
        assert len(image_array) == 2
        #numpy arrays need to be compared with array_equal so can't use Mock.assert_called_with(args*)
        assert numpy.array_equal(image_array[0], image_data[struct_channel]) == True
        assert numpy.array_equal(image_array[1], image_data["638nm"]) == True

    @pytest.mark.parametrize("image_data,struct_channel", 
                             [({"488nm": numpy.array([1, 2, 3])}, "488nm"),
                              ({"561nm": numpy.array([1, 2, 3])}, "561nm")
                             ])
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.Structures")
    @mock.patch("pipeline_qc.segmentation.structure.structure_seg_service.file_processing_methods")
    def test_structure_segmentation_legacy_happy_path(self, mock_file_processing_methods: Mock, mock_structures: Mock, image_data, struct_channel):
        # Arrange
        fov = FovFile(fov_id=63, workflow="Pipeline 4.4", local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456", gene="LMNB1")
        struct_info = StructureInfo(gene="LMNB1", ml=False, ml_model=None, algorithm_name="Python LMNB1 structure segmentation", algorithm_version="1.0.0")
        mock_file_processing_methods.split_image_into_channels.return_value = image_data
        mock_structures.get.return_value = struct_info
        self._mock_repository.segmentation_exists.return_value = False
        self._mock_legacy_segmenter.process_img.return_value = numpy.array([]), numpy.array([])

        # Act     
        result: SegmentationResult = self._structure_seg_service.structure_segmentation(fov, 
                                                                                        save_to_fms=False, 
                                                                                        save_to_filesystem=False, 
                                                                                        output_dir="", 
                                                                                        process_duplicates=False)
        # Assert
        assert result.fov_id == 63
        assert result.status == ResultStatus.SUCCESS
        self._mock_legacy_segmenter.process_img.assert_called_once()
        assert self._mock_legacy_segmenter.process_img.call_args[1]["gene"] == fov.gene
        #numpy arrays need to be compared with array_equal so can't use Mock.assert_called_with(args*)
        assert numpy.array_equal(self._mock_legacy_segmenter.process_img.call_args[1]["image"], image_data[struct_channel]) == True