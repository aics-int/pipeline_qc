from aics_dask_utils import DistributedHandler
from dask_jobqueue import SLURMCluster

"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
import logging
import sys
import traceback
import lkaccess.contexts

from pipeline_qc.image_qc_methods.cell_seg_wrapper import CellSegmentationWrapper
from pipeline_qc.image_qc_methods.cell_seg_uploader import CellSegmentationUploader, FileManagementSystem

###############################################################################

log = logging.getLogger()
# Note: basicConfig should only be called in bin scripts (CLIs).
# https://docs.python.org/3/library/logging.html#logging.basicConfig
# "This function does nothing if the root logger already has handlers configured for it."
# As such, it should only be called once, and at the highest level (the CLIs in this case).
# It should NEVER be called in library code!
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s')

###############################################################################

CONFIG = {
    "prod":{
        "fms_host": "aics.corp.alleninstitute.org",
        "fms_port": 80,
        "fms_timeout_in_seconds": 300,
        "labkey_context": lkaccess.contexts.PROD
    },
    "stg":{
        "fms_host": "stg-aics.corp.alleninstitute.org",
        "fms_port": 80,
        "fms_timeout_in_seconds": 300,
        "labkey_context": lkaccess.contexts.STAGE
    },
    "dev":{
        "fms_host": "dev-aics-ssl-001.corp.alleninstitute.org",
        "fms_port": 8080,
        "fms_timeout_in_seconds": 300,
        "labkey_context": lkaccess.contexts.DEV
    }
}

class Args(argparse.Namespace):

    def __init__(self):
        super().__init__()
        # Arguments that could be passed in through the command line
        self.output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'
        # self.json_dir = '/allen/aics/microscopy/Aditya/image_qc_outputs/json_logs'
        self.workflows = None
        self.cell_lines = None
        self.plates = None
        self.fovids = None
        self.only_from_fms = True
        self.save_to_fms = False
        self.save_to_isilon = False
        self.env = 'stg'
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='Cell and Nuclear Segmentations',
                                    description='Generates Cell and nuclear Segmentations for a series of fovs. '
                                                'Can filter based on workflow, cell line, plate, or specific fovids')
        p.add_argument('--output_dir', type=str,
                       help='directory which all files should be saved',
                       default='/allen/aics/microscopy/Aditya/cell_segmentations', required=False)
        p.add_argument('--workflows', nargs='+',
                       help="Array of workflows to run segmentations on. E.g. --workflows '[PIPELINE 4]' '[PIPELINE 4.4'] ",
                       default=None, required=False)
        p.add_argument('--cell_lines', nargs='+',
                       help="Array of Cell-lines to run segmentations on. E.g. --cell_lines 'AICS-11' 'AICS-7' ",
                       default=None, required=False)
        p.add_argument('--plates', nargs='+',
                       help="Array of plates to run segmentations on. E.g. --plates '3500003813' '3500003642' ",
                       default=None, required=False)
        p.add_argument('--fovids', nargs='+',
                       help="Array of fovids to run segmentations on. E.g. --fovs '123' '6' ",
                       default=None, required=False)
        p.add_argument('--only_from_fms', type=str,
                       help="Boolean to say whether to only run query on data in fms (default is True)",
                            default=True, required=False)
        p.add_argument('--save_to_fms',
                       help="Save segmentations in fms (default is False)",
                       default=False, required=False, action='store_true')
        p.add_argument('--save_to_isilon',
                       help="Save segmentations on the isilon (default is False)",
                       default=False, required=False, action='store_true')
        p.add_argument('--env', type=str,
                       help="Environment that data will be stored to('prod, 'stg', 'dev' (default is 'stg')",
                       default='stg', required=False)
        p.add_argument('--debug',
                       help='Enable debug mode',
                       default=False, required=False, action='store_true')

        p.parse_args(namespace=self)


###############################################################################


def get_app_root(env: str) -> CellSegmentationWrapper:
    """
    Build dependency tree and return application root
    """
    conf = CONFIG[env]
    fms = FileManagementSystem(host=conf["fms_host"], port=conf["fms_port"])
    uploader = CellSegmentationUploader(fms_client=fms, fms_timeout=conf["fms_timeout_in_seconds"])
    return CellSegmentationWrapper(uploader, conf["labkey_context"])

def main():
    args = Args()
    dbg = args.debug

    try:        
        cell_seg: CellSegmentationWrapper = get_app_root(args.env)

        def work(cell_seg_wrapper, fov_id):
            successResponse = f"FOV {fov_id} success"
            failureResponse = f"FOV {fov_id} failure"

            try:
                cell_seg_wrapper.batch_cell_segmentations(
                    output_dir=args.output_dir,
                    fovids=[fov_id],
                    save_to_fms=args.save_to_fms,
                    save_to_isilon=args.save_to_isilon
                )
                
                return successResponse

            except Exception as e:
                print(f"FOV {fov_id}: Exception: {e}")                
                return failureResponse


        # Run distributed
        cluster = SLURMCluster(cores=1, 
                               memory="50GB", 
                               queue="aics_gpu_general", 
                               walltime="00:20:00", 
                               job_extra=["--gres=gpu:gtx1080:1"])   
        cluster.scale(4)

        with DistributedHandler(None) as handler:
            futures = handler.client.map(
                lambda fov_id: work(cell_seg, fov_id),
                args.fovids
            )

        results = handler.gather(futures)
        print("Results:\n")
        for r in results:
            print(f"{r}\n")

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)





                          

if __name__ == "__main__":
    main()