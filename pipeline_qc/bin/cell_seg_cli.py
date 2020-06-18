import argparse
import logging
import sys
import traceback

from datetime import datetime
from pipeline_qc.cell_segmentation.cell_seg_wrapper import CellSegmentationWrapperBase, CellSegmentationWrapper, CellSegmentationDistributedWrapper
from pipeline_qc.cell_segmentation.cell_seg_service import CellSegmentationService
from pipeline_qc.cell_segmentation.cell_seg_repository import CellSegmentationRepository, FileManagementSystem
from pipeline_qc.cell_segmentation.configuration import Configuration, AppConfig, GpuClusterConfig

###############################################################################

# Note: basicConfig should only be called in bin scripts (CLIs).
# https://docs.python.org/3/library/logging.html#logging.basicConfig
# "This function does nothing if the root logger already has handlers configured for it."
# As such, it should only be called once, and at the highest level (the CLIs in this case).
# It should NEVER be called in library code!

log = logging.getLogger()
log.handlers = [] # reset handlers because fnet module sets logging handlers as soon as imported...
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s')

###############################################################################


class Args(argparse.Namespace):

    def __init__(self):
        super().__init__()
        # Arguments that could be passed in through the command line
        self.output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'
        self.workflows = None
        self.cell_lines = None
        self.plates = None
        self.fovids = None
        self.only_from_fms = True
        self.save_to_fms = False
        self.save_to_filesystem = False
        self.env = 'stg'
        self.process_duplicates = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='Cell and Nuclear Segmentations',
                                    description='Generates Cell and nuclear Segmentations for a series of fovs. '
                                                'Can filter based on workflow, cell line, plate, or specific fovids')

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
        p.add_argument('--save_to_filesystem',
                       help="Save segmentations on the filesystem (default is False)",
                       default=False, required=False, action='store_true')
        p.add_argument('--output_dir', type=str,
                       help='directory where files should be saved when saving to filesystem (can be isilon)',
                       default='/allen/aics/microscopy/Aditya/cell_segmentations', required=False)
        p.add_argument('--process_duplicates',
                       help="Re-process segmentation run if existing segmentation is found (default is False)",
                       default=False, required=False, action='store_true')                                              
        p.add_argument('--env', choices=['dev', 'stg', 'prod'],
                       help="Environment that data will be stored to (default is 'stg')",
                       default='stg', required=False)
        p.add_argument('--debug',
                       help='Enable debug mode',
                       default=False, required=False, action='store_true')
        distributed = p.add_argument_group("distributed", "Distributed run options")
        distributed.add_argument('--distributed',
                                 help="Run in distributed mode (default is False). Use with --gpu to specify cluster gpu type.",
                                 default=False, required=False, action='store_true')
        distributed.add_argument('--gpu', choices=["gtx1080", "titanx", "titanxp", "v100"],
                                 help="Cluster GPU type to use for distributed run (default is 'gtx1080')",
                                 default='gtx1080', required=False)

        p.parse_args(namespace=self)


###############################################################################


def get_app_root(args: Args) -> CellSegmentationWrapperBase:
    """
    Build dependency tree and return application root
    """
    env = args.env

    app_config = AppConfig(Configuration.load(f"config/config.{env}.yaml"))
    fms = FileManagementSystem(host=app_config.fms_host, port=app_config.fms_port)
    repository = CellSegmentationRepository(fms, app_config)
    service = CellSegmentationService(repository, app_config)

    if args.distributed:
        gpu = args.gpu
        cluster_config = GpuClusterConfig(gpu, Configuration.load(f"config/cluster.yaml"))
        return CellSegmentationDistributedWrapper(service, app_config, cluster_config)
    else:    
        return CellSegmentationWrapper(service, app_config)


def main():
    args = Args()

    try:
        print(f"[{datetime.now()}] - Start cell_seg_cli")
        print(f"Environment: {args.env}")

        cell_seg = get_app_root(args)
        cell_seg.batch_cell_segmentations(
            output_dir=args.output_dir,
            workflows=args.workflows,
            cell_lines=args.cell_lines,
            plates=args.plates,
            fovids=args.fovids,
            only_from_fms=args.only_from_fms,
            save_to_fms=args.save_to_fms,
            save_to_filesystem=args.save_to_filesystem,
            process_duplicates=args.process_duplicates
        )

        print(f"[{datetime.now()}] - End cell_seg_cli")

    except Exception as e:
        log.error("=============================================")
        if args.debug:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == '__main__':
    main()
