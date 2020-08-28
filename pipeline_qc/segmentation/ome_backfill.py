import aicsimageio
import logging
import traceback

from enum import Enum
from aicsimageio.writers import OmeTiffWriter
from logging import Formatter, StreamHandler, FileHandler
from datetime import datetime
from dataclasses import dataclass
from pipeline_qc.image_qc_methods.query_fovs import query_fovs
from pipeline_qc.segmentation.common.fov_file import FovFile

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

def configure_logging():
    f = Formatter(fmt='[%(asctime)s][%(levelname)s] %(message)s')
    streamHandler = StreamHandler()
    streamHandler.setFormatter(f)
    fileHandler = FileHandler(filename=f"ome_backfill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log", mode="w")
    fileHandler.setFormatter(f)
    log = logging.getLogger() # root logger
    log.handlers = [streamHandler, fileHandler] # overwrite handlers
    log.setLevel(logging.INFO)

def backfill_fov(row) -> BackfillResult:
    try:
        fov_id = row["fovid"]
        source_file_path = row["localfilepath"]
        seg_file_path = row["latest_segmentation_readpath"]
        channel_names = ["nucleus_segmentation", "membrane_segmentation", "nucleus_contour", "membrane_contour"]

        #get segmentation data
        seg_img = aicsimageio.AICSImage(seg_file_path)
        seg_channels = seg_img.get_channel_names()
        seg_pixel_size = str(seg_img.get_physical_pixel_size()) 

        #skip if already done
        if seg_channels == channel_names and seg_pixel_size != (1.0, 1.0, 1.0):
            return BackfillResult(fov_id=fov_id, status=ResultStatus.SKIPPED, message="OME Metadata already present")
                         
        #get source data
        source_img = aicsimageio.AICSImage(source_file_path)
        source_pixel_size = source_img.get_physical_pixel_size()
        
        #save segmentation with ome metadata
        seg_img.data #preload
        data = seg_img.get_image_data("TCZYX")
        with OmeTiffWriter(seg_file_path, overwrite_file=True) as writer:
            writer.save(data=data, channel_names=channel_names,  pixels_physical_size=source_pixel_size, dimension_order="TCZYX")

        return BackfillResult(fov_id=fov_id, status=ResultStatus.SUCCESS)

    except Exception as ex:
        msg = f"Exception: {str(ex)}\n{traceback.format_exc()}"
        return BackfillResult(fov_id=fov_id, status=ResultStatus.FAILED, message=msg)

def main():
    #aicsimageio.use_dask(False)
    configure_logging()
    log = logging.getLogger(__name__)
    log.info("Start: ome_backfill")

    #df = query_fovs(workflows=["Pipeline 4", "Pipeline 4.1", "Pipeline 4.2", "Pipeline 4.4"])
    df = query_fovs(fovids=[154040], labkey_host="stg-aics.corp.alleninstitute.org")

    log.info(f"Backfilling {len(df)} FOVs")

    for index, row in df.iterrows():
        result = backfill_fov(row)
        log.info(result)
    
    log.info("End: ome_backfill")

if __name__ == "__main__":
    main()