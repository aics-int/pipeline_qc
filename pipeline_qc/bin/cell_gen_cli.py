import argparse
import logging
import sys
import traceback

from lkaccess import LabKey, contexts

import pipeline_qc.cell_generation.labkey_cell_generation as cell_generation
from pipeline_qc.image_qc_methods.query_fovs import query_fovs


CONTEXTS_MAP = {
    'prod': contexts.PROD,
    'stg': contexts.STAGE,
    'dev': contexts.DEV
}


def _configure_logging(debug: bool):
    log = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
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
                                    description='Generates LabKey "Cell" entries for a series of fovs using their '
                                                'latest segmentation results.'
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
        p.add_argument('--env', choices=['dev', 'stg', 'prod'],
                       help="Environment that data will be stored to (default is 'stg')",
                       default='stg', required=False)
        p.add_argument('--debug',
                       help='Enable debug mode',
                       default=False, required=False, action='store_true')

        p.parse_args(namespace=self)


def main():
    args = Args()
    _configure_logging(args.debug)
    log = logging.getLogger(__name__)

    try:
        log.info("Start cell_gen_cli")
        log.info(f"Environment: {args.env}")

        lk = LabKey(server_context=CONTEXTS_MAP[args.env])

        log.info("Querying FOV info")
        fovs_df = query_fovs(
            workflows=args.workflows,
            cell_lines=args.cell_lines,
            plates=args.plates,
            fovids=args.fovids,
            labkey_host='stg-aics.corp.alleninstitute.org',  # TODO Use host from env
            labkey_port=80  # TODO: Use port from env
        )
        cell_generation.generate_cells_from_fov_ids(fovs_df, lk)

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

