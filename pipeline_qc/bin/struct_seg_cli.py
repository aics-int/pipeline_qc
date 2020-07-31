import argparse
import logging
import sys
import traceback

from logging import FileHandler, StreamHandler, Formatter
from datetime import datetime
from pipeline_qc.segmentation.structure.structure_seg_wrapper import StructureSegmentationWrapperBase, StructureSegmentationWrapper, StructureSegmentationWrapperDistributed
from pipeline_qc.segmentation.structure.structure_seg_service import StructureSegmentationService, StructureSegmenter
from pipeline_qc.segmentation.structure.structure_seg_repository import StructureSegmentationRepository, FileManagementSystem
from pipeline_qc.segmentation.configuration import Configuration, AppConfig, GpuClusterConfig
from pipeline_qc.segmentation.common.labkey_provider import LabkeyProvider, LabKey

class Args(argparse.Namespace):

    def __init__(self):
        super().__init__()
        # Arguments that could be passed in through the command line
        self.output_dir = '/allen/aics/microscopy/Aditya/structure_segmentations'
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
                       help="Array of workflows to run segmentations on. E.g. --workflows 'Pipeline 4' 'Pipeline 4.4' ",
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

def configure_logging(debug: bool):
    f = Formatter(fmt='[%(asctime)s][%(levelname)s] %(message)s')
    streamHandler = StreamHandler()
    streamHandler.setFormatter(f)
    fileHandler = FileHandler(filename=f"struct_seg_cli_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log", mode="w")
    fileHandler.setFormatter(f)
    log = logging.getLogger() # root logger
    log.handlers = [streamHandler, fileHandler] # overwrite handlers
    log.setLevel(logging.DEBUG if debug else logging.INFO)


def get_app_root(args: Args) -> StructureSegmentationWrapperBase:
    """
    Build dependency tree and return application root
    """
    env = args.env

    app_config = AppConfig(Configuration.load(f"config/config.{env}.yaml"))
    fms = FileManagementSystem(host=app_config.fms_host, port=app_config.fms_port)
    labkey = LabKey(host=app_config.labkey_host, port=app_config.labkey_port)
    labkey_provider = LabkeyProvider(labkey)
    repository = StructureSegmentationRepository(fms, labkey_provider, app_config)
    legacy_segmenter = StructureSegmenter()
    service = StructureSegmentationService(legacy_segmenter, repository, app_config)

    if args.distributed:
        gpu = args.gpu
        cluster_config = GpuClusterConfig(gpu, Configuration.load(f"config/cluster.yaml"))
        return StructureSegmentationWrapperDistributed(service, app_config, cluster_config)
    else:    
        return StructureSegmentationWrapper(service, app_config)


def main():
    args = Args()
    debug = args.debug
    configure_logging(debug)
    log = logging.getLogger(__name__)

    try:
        log.info("Start struct_seg_cli")
        log.info(f"Environment: {args.env}")
        log.info(args)

        struct_seg = get_app_root(args)
        struct_seg.batch_structure_segmentations(
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

        log.info("End struct_seg_cli")

    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == '__main__':
    main()
