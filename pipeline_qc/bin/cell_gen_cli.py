import argparse
import logging
import sys
import traceback

from datetime import datetime
from logging import FileHandler, StreamHandler, Formatter
from lkaccess import LabKey

import pipeline_qc.cell_generation.labkey_cell_generation as cell_generation
from pipeline_qc.image_qc_methods.query_fovs import query_fovs

# TODO: This config should be refactored to live outside the segmentation folder
from pipeline_qc.segmentation.configuration import Configuration, AppConfig

INIT_TIME = datetime.utcnow().strftime('%Y%m%d_%H%M%S')


def _configure_logging(debug: bool):
    f = Formatter(fmt='[%(asctime)s][%(levelname)s] %(message)s')
    streamHandler = StreamHandler()
    streamHandler.setFormatter(f)
    fileHandler = FileHandler(filename=f"cell_gen_cli_{INIT_TIME}.log", mode="w")
    fileHandler.setFormatter(f)
    log = logging.getLogger()  # root logger
    log.handlers = [streamHandler, fileHandler]  # overwrite handlers
    log.setLevel(logging.DEBUG if debug else logging.INFO)


class Args(argparse.Namespace):
    def __init__(self):
        super().__init__()
        # Arguments that could be passed in through the command line
        self.workflows = None
        self.cell_lines = None
        self.plates = None
        self.fovids = None
        self.env = 'stg'
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='LabKey Cell Generation',
                                    description='Generates LabKey "Cell" entries for a series of FOVs using their '
                                                'latest segmentation results.'
                                                'Can filter based on workflow, cell line, plate, or specific FOVIds')

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
        p.add_argument('--env', choices=['dev', 'stg', 'prod'],
                       help="Environment that data will be stored to (default is 'stg')",
                       default='stg', required=False)
        p.add_argument('--debug',
                       help='Enable debug mode',
                       default=False, required=False, action='store_true')

        p.parse_args(namespace=self)


def main():
    args = Args()
    app_config = AppConfig(Configuration.load(f"config/config.{args.env}.yaml"))

    _configure_logging(args.debug)
    log = logging.getLogger(__name__)

    try:
        log.info("Start cell_gen_cli")
        log.info(f"Environment: {args.env}")

        lk = LabKey(
            host=app_config.labkey_host,
            port=app_config.labkey_port
        )

        log.info("Querying FOV info")
        fovs_df = query_fovs(
            workflows=args.workflows,
            cell_lines=args.cell_lines,
            plates=args.plates,
            fovids=args.fovids,
            labkey_host=app_config.labkey_host,
            labkey_port=app_config.labkey_port
        )
        cell_generation.generate_cells_from_fov_ids(fovs_df, lk, init_time=INIT_TIME)

        log.info("End cell_gen_cli")

    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


if __name__ == '__main__':
    main()

