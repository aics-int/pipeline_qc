import pytest

from unittest.mock import Mock, call
from pipeline_qc.segmentation.structure.structure_seg_repository import LabkeyProvider, FileManagementSystem, StructureSegmentationRepository, StructureInfo, RunInfo
from pipeline_qc.segmentation.configuration import AppConfig

class TestStructureSegmentationRepository:
    @pytest.fixture(autouse=True)
    def setup(self):
        self._mock_fms_client = Mock(spec=FileManagementSystem)
        self._mock_labkey_provider = Mock(spec=LabkeyProvider)
        self._struct_seg_repository = StructureSegmentationRepository(fms_client=self._mock_fms_client, 
                                                                      labkey_provider=self._mock_labkey_provider, 
                                                                      config=Mock(spec=AppConfig))

    def test_upload_structure_segmentation_no_initial_metadata(self): 
        # Arrange
        input_file_id = "abcd1234"
        seg_path = "test/struct_seg.tiff"
        contour_path = "test/struct_contour.tiff"
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name="ML H2B Structure Segmentation", algorithm_version="0.1.0")
        self._mock_fms_client.query_files.return_value = None
        self._mock_labkey_provider.create_run_id.return_value = 1234

        # Act
        self._struct_seg_repository.upload_structure_segmentation(struct_info, input_file_id, seg_path, contour_path)

        # Assert
        assert self._mock_fms_client.upload_file_sync.call_count == 2
        contour_upload_path = self._mock_fms_client.upload_file_sync.call_args_list[0][0][0]
        contour_metadata = self._mock_fms_client.upload_file_sync.call_args_list[0][0][1]
        seg_upload_path = self._mock_fms_client.upload_file_sync.call_args_list[1][0][0]
        seg_metadata = self._mock_fms_client.upload_file_sync.call_args_list[1][0][1]

        assert contour_upload_path == contour_path
        assert contour_metadata["file"]["file_type"] == "image"
        assert len(contour_metadata["content_processing"]["channels"]) == 1
        assert contour_metadata["content_processing"]["channels"]["0"]["content_type"] == "Structure contour"
        assert contour_metadata["content_processing"]["channels"]["0"]["run_id"] == 1234
        assert contour_metadata["provenance"]["input_files"][0] == input_file_id
        assert contour_metadata["provenance"]["algorithm"] == struct_info.algorithm_name
        assert contour_metadata["provenance"]["algorithm_version"] == struct_info.algorithm_version   

        assert seg_upload_path == seg_path        
        assert seg_metadata["file"]["file_type"] == "image"
        assert len(seg_metadata["content_processing"]["channels"]) == 1
        assert seg_metadata["content_processing"]["channels"]["0"]["content_type"] == "Structure segmentation"
        assert seg_metadata["content_processing"]["channels"]["0"]["run_id"] == 1234
        assert seg_metadata["provenance"]["input_files"][0] == input_file_id
        assert seg_metadata["provenance"]["algorithm"] == struct_info.algorithm_name
        assert seg_metadata["provenance"]["algorithm_version"] == struct_info.algorithm_version 
        
    def test_upload_structure_segmentation_with_initial_metadata(self): 
        # Arrange
        input_file_id = "abcd1234"
        seg_path = "test/struct_seg.tiff"
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name="ML H2B Structure Segmentation", algorithm_version="0.1.0")
        self._mock_fms_client.query_files.return_value = [{"microscopy": {"fov_id": "9999"}}]
        self._mock_labkey_provider.create_run_id.return_value = 1234

        # Act
        self._struct_seg_repository.upload_structure_segmentation(struct_info, input_file_id, seg_path)

        # Assert
        metadata = self._mock_fms_client.upload_file_sync.call_args[0][1]
        assert metadata["microscopy"]["fov_id"] == "9999" 

    def test_segmentation_exists_found_single_file(self):
         # Arrange
        algorithm = "ML H2B Structure Segmentation"
        algorithm_version = "0.1.0"
        self._mock_fms_client.query_files.return_value = [
            {"file_id": "1234", "file_name": "struct.tiff", "content_processing": {"channels": {"0": {"run_id": 1234, "algorithm": algorithm, "algorithm_version": algorithm_version}}}}
        ]      
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name=algorithm, algorithm_version=algorithm_version)

        # Act
        exists = self._struct_seg_repository.segmentation_exists("struct.tiff", struct_info)

        # Assert
        assert exists == True
    
    def test_segmentation_exists_found_multiple_files(self):
         # Arrange
        algorithm = "ML H2B Structure Segmentation"
        algorithm_version = "0.1.0"
        self._mock_fms_client.query_files.return_value = [
            {"file_id": "1234", "file_name": "struct.tiff","content_processing": {"channels": {"0": {"run_id": 1234}}}},
            {"file_id": "1234", "file_name": "struct.tiff","content_processing": {"channels": {"0": {"run_id": 5678}}}},
            {"file_id": "1234", "file_name": "struct.tiff","content_processing": {"channels": {"0": {"run_id": 9999, "algorithm": algorithm, "algorithm_version": algorithm_version}}}}
        ]      
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name=algorithm, algorithm_version=algorithm_version)

        # Act
        exists = self._struct_seg_repository.segmentation_exists("struct.tiff", struct_info)

        # Assert
        assert exists == True        
    
    @pytest.mark.parametrize("result", [None, []])
    def test_segmentation_exists_file_not_found(self, result):
         # Arrange
        self._mock_fms_client.query_files.return_value = result      
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name="ML H2B Structure Segmentation", algorithm_version="0.1.0")

        # Act
        exists = self._struct_seg_repository.segmentation_exists("struct.tiff", struct_info)

        # Assert
        assert exists == False

    def test_segmentation_exists_no_algorithm_info(self):
         # Arrange
        self._mock_fms_client.query_files.return_value = [{"file_id": "1234", "file_name": "struct.tiff"}]      
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name="ML H2B Structure Segmentation", algorithm_version="0.1.0")

        # Act
        exists = self._struct_seg_repository.segmentation_exists("struct.tiff", struct_info)

        # Assert
        assert exists == False

    def test_segmentation_exists_mismatch_algorithm_info(self):
         # Arrange
        self._mock_fms_client.query_files.return_value = [
            {"file_id": "1234", "file_name": "struct.tiff", "content_processing": {"channels": {"0": {"run_id": 1234, "algorithm": "old algorithm", "algorithm_version": "0.0.0"}}}},
            {"file_id": "1234", "file_name": "struct.tiff", "content_processing": {"channels": {"0": {"run_id": 1234, "algorithm": "old algorithm", "algorithm_version": "0.1.0"}}}},
            {"file_id": "1234", "file_name": "struct.tiff", "content_processing": {"channels": {"0": {"run_id": 1234, "algorithm": "ML H2B Structure Segmentation", "algorithm_version": "0.1.1"}}}}
        ]      
        struct_info = StructureInfo(gene="H2B", ml=True, ml_model="structure_H2B_production", algorithm_name="ML H2B Structure Segmentation", algorithm_version="0.1.0")

        # Act
        exists = self._struct_seg_repository.segmentation_exists("struct.tiff", struct_info)

        # Assert
        assert exists == False