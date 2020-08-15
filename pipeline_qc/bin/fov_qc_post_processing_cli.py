"""
This script will allow us to call the fov_qc_post_processing script through a cli
"""

import argparse
import logging
import sys
import traceback

from pipeline_qc.image_qc_methods import fov_qc_post_processing

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
        self.env = 'stg'
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='FOV QC Post Processing',
                                    description='Generates pass fail criteria from FOV QC metrics')
        p.add_argument('--env', type=str,
                       help="Environment that data will be stored to('prod, 'stg', default is 'stg')",
                       default='stg', required=False)
        p.add_argument('--debug',
                       help='Enable debug mode',
                       default=False, required=False, action='store_true')

        p.parse_args(namespace=self)


###############################################################################

def main():
    args = Args()
    dbg = args.debug

    try:
        fov_qc_post_processing.update_qc_data_labkey(
            df=fov_qc_post_processing.z_score_stat_generation(),
            env = args.env
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
