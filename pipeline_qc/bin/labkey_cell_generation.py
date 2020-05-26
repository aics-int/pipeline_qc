import pipeline_qc.labkey_cell_generation as cell_generation

import argparse
import json
import logging
import sys
import traceback

from lkaccess import LabKey, contexts

log = logging.getLogger('bin/labkey_cell_generation')
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s')
logging.getLogger('labkey').setLevel(logging.ERROR)

CONTEXTS_MAP = {
    'prod': contexts.PROD,
    'stage': contexts.STAGE,
    'dev': contexts.DEV
}


class Args(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.metadata = None
        self.lk_context = None
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            description='Inserts rows in the LabKey Processing.Cell for each cell present in the given segmentation'
                        ' output file.'
        )

        p.add_argument('-f', '--file', dest='input_file', required=True,
                       help='Path to a segmentation file to be used as input')
        p.add_argument('-md', '--metadata', dest='metadata', required=True,
                       help='Path to a .json file containing the metadata for the segmentation file')
        p.add_argument('-lk', '--labkey-context', dest='lk_context', required=True,
                       choices=['prod', 'stage', 'dev'],
                       help='LabKey environment to pull data from and insert data into.')

        p.parse_args(namespace=self)


def main():
    try:
        args = Args()

        input_file = args.input_file

        with open(args.metadata) as json_file:
            metadata = json.load(json_file)
        lk = LabKey(server_context=CONTEXTS_MAP[args.lk_context])

        cell_generation.generate_cells(input_file, metadata, lk)

    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


if __name__ == '__main__':
    main()
