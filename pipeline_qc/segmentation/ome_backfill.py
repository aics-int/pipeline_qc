import aicsimageio
import logging
import traceback
import os
import json

from tempfile import TemporaryDirectory
from enum import Enum
from aicsimageio.writers import OmeTiffWriter
from logging import Formatter, StreamHandler, FileHandler
from datetime import datetime
from dataclasses import dataclass
from aicsfiles import FileManagementSystem
from pipeline_qc.image_qc_methods.query_fovs import query_fovs
from pipeline_qc.segmentation.common.fov_file import FovFile

LK_HOST = "stg-aics.corp.alleninstitute.org"
FMS_HOST = "stg-aics.corp.alleninstitute.org"

class ResultStatus(Enum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass
class BackfillResult:
    fov_id: int
    status: ResultStatus
    message: str = ""

    def __str__(self):
        return f"FOV {self.fov_id}: {self.status.value}\n{self.message}"

class BackfillService:
    def __init__(self):
        self._fms = FileManagementSystem(host=FMS_HOST)
        self.log = logging.getLogger(__name__)

    def run_backfill(self):
        #df = query_fovs(workflows=["Pipeline 4", "Pipeline 4.1", "Pipeline 4.2", "Pipeline 4.4"])
        df = query_fovs(fovids=[154040], labkey_host=LK_HOST)

        self.log.info(f"Backfilling {len(df)} FOVs")

        with TemporaryDirectory() as tmpdir:
            for index, row in df.iterrows():
                result = self._backfill_fov(row, tmpdir)
                self.log.info(result)

    def _backfill_fov(self, row, tmpdir) -> BackfillResult:
        try:
            fov_id = row["fovid"]
            source_file_path = row["localfilepath"]
            seg_read_path = row["latest_segmentation_readpath"]
            seg_metadata = json.loads(row["latest_segmentation_metadata"])
            seg_file_name = os.path.basename(seg_read_path)
            assert seg_file_name == seg_metadata["file"]["file_name"]
            channel_names = ["nucleus_segmentation", "membrane_segmentation", "nucleus_contour", "membrane_contour"]
            

            #get segmentation img metadata
            seg_img = aicsimageio.AICSImage(seg_read_path)
            seg_channels = seg_img.get_channel_names()
            seg_pixel_size = str(seg_img.get_physical_pixel_size()) 

            #skip if already done
            if seg_channels == channel_names and seg_pixel_size != (1.0, 1.0, 1.0):
                return BackfillResult(fov_id=fov_id, status=ResultStatus.SKIPPED, message="OME Metadata already present")
                            
            #get source img metadata
            source_img = aicsimageio.AICSImage(source_file_path)
            source_pixel_size = source_img.get_physical_pixel_size()
            
            #save segmentation with new ome metadata
            seg_img.data #preload
            data = seg_img.get_image_data("TCZYX")           
            upload_file_path = f"{tmpdir}/{seg_file_name}"
            with OmeTiffWriter(upload_file_path, overwrite_file=True) as writer:
                writer.save(data=data, channel_names=channel_names,  pixels_physical_size=source_pixel_size, dimension_order="TCZYX")

            #upload new version to FMS
            seg_metadata.update({"file": {"file_type": "image"}}) # clear old file block, just in case
            self._fms.upload_file(file=upload_file_path, metadata=seg_metadata, timeout=120)

            return BackfillResult(fov_id=fov_id, status=ResultStatus.SUCCESS)

        except Exception as ex:
            msg = f"Exception: {str(ex)}\n{traceback.format_exc()}"
            return BackfillResult(fov_id=fov_id, status=ResultStatus.FAILED, message=msg)


def configure_logging():
    f = Formatter(fmt='[%(asctime)s][%(levelname)s] %(message)s')
    streamHandler = StreamHandler()
    streamHandler.setFormatter(f)
    fileHandler = FileHandler(filename=f"ome_backfill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log", mode="w")
    fileHandler.setFormatter(f)
    log = logging.getLogger() # root logger
    log.handlers = [streamHandler, fileHandler] # overwrite handlers
    log.setLevel(logging.INFO)

def main():
    #aicsimageio.use_dask(False)
    configure_logging()
    log = logging.getLogger(__name__)
    log.info("Start: ome_backfill")

    BackfillService().run_backfill()
    
    log.info("End: ome_backfill")

if __name__ == "__main__":
    main()