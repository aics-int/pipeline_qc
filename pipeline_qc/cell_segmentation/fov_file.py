from dataclasses import dataclass
from pandas import Series

@dataclass
class FovFile:
    """
    Represents a FOV file record
    """

    fov_id: int
    workflow: str
    local_file_path: str
    source_image_file_id: str

    @property
    def is_single_camera(self) -> bool:
        """
        Indicate whether this FOV was image with a single camera (True) or dual camera (False) 
        """
        return self.workflow.lower() in ["pipeline 4",
                                         "pipeline 4.1", 
                                         "pipeline 4.2"
                                        ]

    @staticmethod
    def from_dataframe_row(row: Series):
        """
        Map FovFile from a fov query dataframe row
        """
        return FovFile(fov_id=row["fovid"],
                       workflow=row["workflow"][0],
                       local_file_path=row["localfilepath"],
                       source_image_file_id=row["sourceimagefileid"]
                       )