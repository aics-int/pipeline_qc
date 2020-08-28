import aicsimageio
import logging
import traceback
import os
import json
import sys

from aics_dask_utils import DistributedHandler
from dask.distributed import LocalCluster, as_completed, Client
from dask_jobqueue import SLURMCluster
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

        for index, row in df.iterrows():
            result = self._backfill_fov(row)
            self.log.info(result)

    def run_backfill_distributed(self):
        #df = query_fovs(workflows=["Pipeline 4", "Pipeline 4.1", "Pipeline 4.2", "Pipeline 4.4"])
        df = query_fovs(cell_lines=["AICS-10"], labkey_host=LK_HOST)
        self.log.info(f"Backfilling {len(df)} FOVs")

        rows = []
        for index, row in df.iterrows():
            rows.append(row)

        #cluster = LocalCluster()
        cluster = SLURMCluster(cores=1, queue="aics_cpu_general", walltime="9-24:00:00", nanny=True, memory="1G")
        cluster.scale(25)
        client = Client(cluster)


        futures = client.map(
                lambda row: self._backfill_fov(row),
                rows
        )

        self.log.info("Results:")
        for future, result in as_completed(futures, with_results=True):
            self.log.info(result)

        cluster.close()

    def _backfill_fov(self, row) -> BackfillResult:
        aicsimageio.use_dask(False)

        try:
            fov_id = row["fovid"]
            source_file_path = row["localfilepath"]
            seg_read_path = row["latest_segmentation_readpath"]
            channel_names = ["nucleus_segmentation", "membrane_segmentation", "nucleus_contour", "membrane_contour"]
            
            #skip if no segmentation
            if seg_read_path is None or seg_read_path == "":
                return BackfillResult(fov_id=fov_id, status=ResultStatus.SKIPPED, message="Segmentation not found")

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
            seg_metadata = json.loads(row["latest_segmentation_metadata"])
            seg_file_name = os.path.basename(seg_read_path)
            assert seg_file_name == seg_metadata["file"]["file_name"]

            seg_img.data #preload
            data = seg_img.get_image_data("TCZYX", S=0)        

            with TemporaryDirectory() as tmpdir:
                upload_file_path = f"{tmpdir}/{seg_file_name}"
                with OmeTiffWriter(upload_file_path, overwrite_file=True) as writer:
                    writer.save(data=data, channel_names=channel_names,  pixels_physical_size=source_pixel_size, dimension_order="TCZYX")

                #upload new version to FMS
                seg_metadata.update({"file": {"file_type": "image"}}) # clear old file block, just in case
                self._fms.upload_file_sync(file_path=upload_file_path, metadata=seg_metadata, timeout=120)

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
    configure_logging()
    log = logging.getLogger(__name__)


    try:
        log.info("Start: ome_backfill")
        #BackfillService().run_backfill()
        BackfillService().run_backfill_distributed()
        log.info("End: ome_backfill")
    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)



if __name__ == "__main__":
    main()