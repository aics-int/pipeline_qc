import pandas as pd
from pipeline_qc import detect_edge, detect_z_stack_false_clip
from pipeline_qc.image_qc_methods import query_fovs, file_processing_methods, intensity
import json
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import argparse

import argparse
import logging
import sys
import traceback

from pipeline_qc import get_module_version, Example

class BatchFovQc:


    def process_single_fov(index, row, json_dir, output_dir, image_gen=False):

        # Splits 6D image into single channel images for qc algorithm processing
        channel_dict = file_processing_methods.split_image_into_channels(row['localfilepath'],
                                                                         str(row['sourceimagefileid']))

        # Initializes a dictionary where all stats for an fov are saved
        stat_dict = dict()

        # Iterates over each z-stack image and runs qc_algorithms, and then adds each stat generated to the stat_dict
        for channel_name, channel_array in channel_dict.items():

            # Runs the intensity metrics on all z-stack images. Put here since run on all channels
            intensity_dict = intensity.intensity_stats_single_channel(channel_array)
            for intensity_key, intensity_value, in intensity_dict.items():
                stat_dict.update({channel_name + ' ' + intensity_key + '-intensity': intensity_value})

            # Runs all metrics to be run on brightfield (edge detection, false clip bf) and makes bf qc_images
            if channel_name == 'brightfield':
                bf_edge_detect = detect_edge.detect_edge_position(channel_array)
                for edge_key, edge_value in bf_edge_detect.items():
                    stat_dict.update({channel_name + ' ' + edge_key: edge_value})
                bf_false_clip_dict = detect_z_stack_false_clip.detect_false_clip_bf(channel_array)
                for false_clip_key, false_clip_value in bf_false_clip_dict.items():
                    stat_dict.update({channel_name + ' ' + false_clip_key + '-false clip': false_clip_value})

                # PUT QC_IMAGES FOR BF HERE
                if image_gen:
                    file_processing_methods.generate_qc_images(channel_array, output_dir, row['fovid'], channel_name)

            # Runs all metrics to be run on 638 (false clip 638) and makes 638 qc_images
            elif channel_name == '638nm':
                bf_false_clip_dict = detect_z_stack_false_clip.detect_false_clip_cmdr(channel_array)
                for false_clip_key, false_clip_value in bf_false_clip_dict.items():
                    stat_dict.update({channel_name + ' ' + false_clip_key + '-false clip': false_clip_value})

                # PUT QC_IMAGES FOR 638 HERE
                if image_gen:
                    file_processing_methods.generate_qc_images(channel_array, output_dir, row['fovid'], channel_name)

            with open(f"{json_dir}/{row['fovid']}.json", "w") as write_out:
                json.dump(stat_dict, write_out)
            file_processing_methods.insert_qc_data_labkey(row['fovid'], stat_dict)
            return stat_dict


    def batch_qc(output_dir, json_dir, workflows=[], cell_lines=[], plates=[], fovids=[], only_from_fms=True, image_gen=False):
        # Runs qc steps and collates all data into a single dataframe for easy sorting and plotting
        # Runs on multiple files, to be used with the query_fms function
        pd.options.mode.chained_assignment = None

        # Run the query fn on specified cell line
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids, only_from_fms=only_from_fms)

        stat_list = list()

        slurm_cluster = SLURMCluster(
            cores=1,
            memory="4GB",
            queue="aics-cpu-general",
            walltime="10:00:00",
            local_directory={},
            log_directory={},
        )
        slurm_cluster.scale_up(80)

        client = Client(slurm_cluster)

        futures = client.map(
            process_single_fov,
            # Create lists of equal length for each input to process_single_fov
            # Creates two lists, one of index, and one of query_df's contents
            [*zip(*list(query_df.iterrows)),
            #  Creates lists the same lemngth as query_df for json_dir, output_dir, and image_gen, all with the same value
            [json_dir] * len(query_df),
            [output_dir] * len(query_df),
            [False] * len(query_df)]
        )

        # # Iterates through all fovs identifies by query dataframe
        # for index, row in query_df.iterrows():
        #
        #     # Splits 6D image into single channel images for qc algorithm processing
        #     channel_dict = file_processing_methods.split_image_into_channels(row['localfilepath'], str(row['sourceimagefileid']))
        #
        #     # Initializes a dictionary where all stats for an fov are saved
        #     stat_dict = dict()
        #
        #     # Iterates over each z-stack image and runs qc_algorithms, and then adds each stat generated to the stat_dict
        #     for channel_name, channel_array in channel_dict.items():
        #
        #         # Runs the intensity metrics on all z-stack images. Put here since run on all channels
        #         intensity_dict = intensity.intensity_stats_single_channel(channel_array)
        #         for intensity_key, intensity_value, in intensity_dict.items():
        #             stat_dict.update({channel_name + ' ' + intensity_key + '-intensity': intensity_value})
        #
        #         # Runs all metrics to be run on brightfield (edge detection, false clip bf) and makes bf qc_images
        #         if channel_name == 'brightfield':
        #             bf_edge_detect = detect_edge.detect_edge_position(channel_array)
        #             for edge_key, edge_value in bf_edge_detect.items():
        #                 stat_dict.update({channel_name + ' ' + edge_key: edge_value})
        #             bf_false_clip_dict = detect_z_stack_false_clip.detect_false_clip_bf(channel_array)
        #             for false_clip_key, false_clip_value in bf_false_clip_dict.items():
        #                 stat_dict.update({channel_name + ' ' + false_clip_key + '-false clip': false_clip_value})
        #
        #             # PUT QC_IMAGES FOR BF HERE
        #             if image_gen:
        #                 file_processing_methods.generate_qc_images(channel_array, output_dir, row['fovid'], channel_name)
        #
        #         # Runs all metrics to be run on 638 (false clip 638) and makes 638 qc_images
        #         elif channel_name == '638nm':
        #             bf_false_clip_dict = detect_z_stack_false_clip.detect_false_clip_cmdr(channel_array)
        #             for false_clip_key, false_clip_value in bf_false_clip_dict.items():
        #                 stat_dict.update({channel_name + ' ' + false_clip_key + '-false clip': false_clip_value})
        #
        #             # PUT QC_IMAGES FOR 638 HERE
        #             if image_gen:
        #                 file_processing_methods.generate_qc_images(channel_array, output_dir, row['fovid'], channel_name)
        #
        #     # Adds stat_dict to a list of dictionaries, which corresponds to the query_df.
        #     stat_list.append(stat_dict)
        #     print(f"Added {str(row['fovid'])} to stat dictionary")

        stat_list = client.gather(futures)

        # Joins query_df to stat_list, and then writes out a csv of all the data to an output folder
        result = pd.concat([query_df, pd.DataFrame(stat_list)], axis=1)
        result.to_csv(output_dir + '/fov_qc_metrics.csv')

        return result

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

    DEFAULT_FIRST = 10
    DEFAULT_SECOND = 20

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.first = self.DEFAULT_FIRST
        self.second = self.DEFAULT_SECOND
        self.debug = False
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='run_exmaple',
                                    description='A simple example of a bin script')
        p.add_argument('-v', '--version', action='version', version='%(prog)s ' + get_module_version())
        p.add_argument('-f', '--first', action='store', dest='first', type=int, default=self.first,
                       help='The first argument value')
        p.add_argument('-s', '--second', action='store', dest='second', type=int, default=self.second,
                       help='The first argument value')
        p.add_argument('--debug', action='store_true', dest='debug', help=argparse.SUPPRESS)
        p.parse_args(namespace=self)


###############################################################################

def main():
    try:
        args = Args()
        dbg = args.debug

        # Do your work here - preferably in a class or function,
        # passing in your args. E.g.
        batcher = BatchFovQc()
        batcher.batch_qc(args.output_dir, args.workflows, args.cell_lines)

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
