import os
import argparse
import logging
import sys
import traceback

import numpy as np
import yaml

from skimage.io import imsave
from aicsimageio import AICSImage

from simple_utils import get_module_version, Example
from simple_utils.QC_func import simple_zstack_QC
from simple_utils.Proc_func import simple_zstack_Proc
from simple_utils.data_loader import load_filenames


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
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='simple_run',
                                    description='run simple data munging and qc tool')
        #p.add_argument('-v', '--version', action='version', version='%(prog)s ' + get_module_version())
        p.add_argument('--config', required=True, help='path to the configuration file')
        p.add_argument('-d','--debug', action='store_true', dest='debug', help=argparse.SUPPRESS)
        p.parse_args(namespace=self)


###############################################################################

def main():

    args = Args()

    try:

        config = yaml.load(open(args.config, 'r'))

        all_files, timelapse_flag = load_filenames(config['Data'])
        print(all_files)

        if 'QC' in config:
            if not timelapse_flag:
                simple_zstack_QC(all_files, config['QC'])
            else:
                #TODO
                pass

        if 'Proc' in config:
            if not timelapse_flag:
                simple_zstack_Proc(all_files, config['Proc'])
            else:
                #TODO
                pass


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