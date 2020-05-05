"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
import logging
import sys
import traceback

from pipeline_qc.image_qc_methods import cardio_mip_qc_generation

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
        self.plates = None
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='Cardio QC mip generation',
                                    description='Generates Max intensity projects (MIPs) for the cardio pipeline')
        p.add_argument('--plates', nargs='+',
                       help="Array of plates to run qc on. E.g. --plates '3500003813' '3500003642' ",
                       default=None, required=False)
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
        cardio_mip_qc_generation.batch_cardio_qc(plates=args.plates)

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
