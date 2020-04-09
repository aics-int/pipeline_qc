"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
import logging
import sys
import traceback

from pipeline_qc.image_qc_methods import batch_fov_qc

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


class Args(argparse.Namespace):

    def __init__(self):
        super().__init__()
        # Arguments that could be passed in through the command line
        self.output_dir = '/allen/aics/microscopy/Aditya/image_qc_outputs'
        self.json_dir = '/allen/aics/microscopy/Aditya/image_qc_outputs/json_logs'
        self.workflows = None,
        self.cell_lines = None,
        self.plates = None,
        self.fovids = None,
        self.only_from_fms = True
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='FOV QC',
                                    description='Generates FOV QC metrics for a series of fovs. Can filter based on '
                                                'workflow, cell line, plate, or specific fovids')
        p.add_argument('--output_dir', type=str,
                       help='directory which all files should be saved',
                       default='/allen/aics/microscopy/Aditya/image_qc_outputs', required=True)
        p.add_argument('--json_dir', type=str,
                       help='directory which json files for individual fovs',
                       default='/allen/aics/microscopy/Aditya/image_qc_outputs/json_logs', required=False)
        p.add_argument('--workflows', type=str,
                       help="Array of workflows to run qc on. E.g. ['[PIPELINE 4]', '[PIPELINE 4.4'] ",
                       default=None, required=False)
        p.add_argument('--cell_lines', type=str,
                       help="Array of Cell-lines to run qc on. E.g. ['AICS-11', 'AICS-7'] ",
                       default=None, required=False)
        p.add_argument('--plates', type=str,
                       help="Array of plates to run qc on. E.g. ['3500003813', '3500003642'] ",
                       default=None, required=False)
        p.add_argument('--fovids', type=str,
                       help="Array of fovids to run qc on. E.g. ['123', '6'] ",
                       default=None, required=False)
        p.add_argument('--only_from_fms', type=str,
                       help="Boolean to say whether to only run query on data in fms (default is True)",
                            default=True, required=False)
        p.add_argument('--debug',
                       help='Enable debug mode',
                       default=False, required=False, action='store_true')

        p.parse_args(namespace=self)


###############################################################################

def main():
    args = Args()
    dbg = args.debug

    try:
        # Do your work here - preferably in a class or function,
        # passing in your args. E.g.
        batch_fov_qc.batch_qc(
            output_dir=args.output_dir,
            json_dir=args.json_dir,
            workflows=args.workflows,
            cell_lines=args.cell_lines,
            plates=args.plates,
            fovids=args.fovids,
            only_from_fms=args.only_from_fms
        )

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == '__main__':
    main()
