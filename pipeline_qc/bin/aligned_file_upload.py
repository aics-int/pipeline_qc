import pipeline_qc.upload_aligned_files_to_fms as uploader

import argparse
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
        self.input_csv = None
        self.aligned_files_folder = None
        self.lk_context = None
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            description='Inserts rows in the LabKey Processing.Cell for each cell present in the given segmentation'
                        ' output file.'
        )

        p.add_argument('-f', '--file', dest='input_csv', required=True,
                       help='Path to a .csv file containing alignment info')
        p.add_argument('-d', '--directory', dest='aligned_files_folder', required=True,
                       help='Path to a folder containing files to be uploaded')
        p.add_argument('-lk', '--labkey-context', dest='lk_context', required=True,
                       choices=['prod', 'stage', 'dev'],
                       help='LabKey environment to pull data from and insert data into.')

        p.parse_args(namespace=self)


def main():
    try:
        args = Args()

        input_csv = args.input_csv
        folder = args.aligned_files_folder
        lk = LabKey(server_context=CONTEXTS_MAP[args.lk_context])

        uploader.upload_aligned_files(lk, input_csv, folder)

    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


if __name__ == '__main__':
    main()
