import argparse
import json

import pandas as pd
from aicsimageio import dask_utils
from pipeline_qc import detect_edge, detect_z_stack_false_clip
from pipeline_qc.image_qc_methods import (file_processing_methods, intensity,
                                          query_fovs)
from tqdm import tqdm


def process_single_fov(row, json_dir, output_dir, image_gen=False, env='stg'):

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

        # TODO: Need to figure out how to make numpy array json serializable in dict
        # with open(f"{json_dir}/{row['fovid']}.json", "w") as write_out:
        #     json.dump(stat_dict, write_out)
        file_processing_methods.insert_qc_data_labkey(row['fovid'], stat_dict, env)
        return stat_dict


def batch_qc(output_dir, json_dir, workflows=None, cell_lines=None, plates=None, fovids=None, only_from_fms=True, image_gen=False, env='stg'):
    # Runs qc steps and collates all data into a single dataframe for easy sorting and plotting
    # Runs on multiple files, to be used with the query_fms function
    pd.options.mode.chained_assignment = None

    # Run the query fn on specified cell line
    query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids, only_from_fms=only_from_fms)
    print(f'''
    __________________________________________

    {len(query_df)} fovs were found to process.
    
    __________________________________________
    ''')

    # Spawn local cluster to speed up image read
    with dask_utils.cluster_and_client() as (cluster, client):

        # Collect all stats
        stat_list = [process_single_fov(row, json_dir, output_dir, False) for i, row in tqdm(query_df.iterrows())]

        image_gen * len(query_df),
    # Joins query_df to stat_list, and then writes out a csv of all the data to an output folder
    result = pd.concat([query_df, pd.DataFrame(stat_list)], axis=1)
    result.to_csv(output_dir + '/fov_qc_metrics.csv')

    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='directory which all files should be saved', required=True)
    parser.add_argument('--json_dir', type=str, help='directory which json files for individual fovs', default = '/allen/aics/microscopy/Aditya/image_qc_outputs/json_logs', required=False)
    parser.add_argument('--workflows', type=str, help="Array of workflows to run qc on. E.g. ['PIPELINE4' 'PIPELINE4.4'] ",default = None, required=False)
    parser.add_argument('--cell_lines', type=str, help="Array of Cell-lines to run qc on. E.g. 'AICS-11' 'AICS-7' ", default = None, required=False)
    parser.add_argument('--plates', type=str, help="Array of plates to run qc on. E.g. '3500003813' '3500003642' ", default = None, required=False)
    parser.add_argument('--fovids', type=str, help="Array of fovids to run qc on. E.g. '123' '6' ", default = None, required=False)
    parser.add_argument('--only_from_fms', type=str, help="Boolean to say whether to only run query on data in fms (default is True)", default=True, required=False)

    args = parser.parse_args()

    batch_qc(
        output_dir=args.output_dir,
        json_dir=args.json_dir,
        workflows=args.workflows,
        cell_lines=args.cell_lines,
        plates=args.plates,
        fovids=args.fovids,
        only_from_fms=args.only_from_fms
    )


if __name__ == '__main__':
    main()
