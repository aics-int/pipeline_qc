import pytest

from pandas import Series
from pipeline_qc.cell_segmentation.fov_file import FovFile

class TestFovFile:

    def test_from_dataframe_row(self):
        # Arrange
        expected_fov_id = 63
        expected_workflow = "Pipeline 4.4"
        expected_local_path = "/allen/aics/some/place/file.tiff"
        expected_source_file_id = "abcdef123456"
        row = Series(index=["fovid", "workflow", "localfilepath", "sourceimagefileid"],
                     data=[expected_fov_id, [expected_workflow], expected_local_path, expected_source_file_id])
        
        # Act
        fov = FovFile.from_dataframe_row(row)

        # Assert
        assert fov.fov_id == expected_fov_id
        assert fov.workflow == expected_workflow
        assert fov.local_file_path == expected_local_path
        assert fov.source_image_file_id == expected_source_file_id


    @pytest.mark.parametrize("workflow,expected", 
                             [("Pipeline 4", True), 
                              ("Pipeline 4.1", True),
                              ("Pipeline 4.2", True),
                              ("Pipeline 4.3", True),
                              ("Pipeline 4.4", False),
                              ("Pipeline 4.5", False),
                              ("Pipeline 4.6", False),
                              ("Pipeline 4.7", False)
                             ])
    def test_is_single_camera(self, workflow: str, expected: bool):
        # Act
        fov = FovFile(fov_id=63, workflow=workflow, local_file_path="/allen/aics/some/place/file.tiff", source_image_file_id="abcdef123456")

        # Assert
        assert fov.is_single_camera == expected