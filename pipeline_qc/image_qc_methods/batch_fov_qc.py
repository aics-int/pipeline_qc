import pickle
import os
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from aics_dask_utils import DistributedHandler
from dask_jobqueue import SLURMCluster
import dask.config
from dask_jobqueue import SLURMCluster
from pipeline_qc import detect_edge, detect_z_stack_false_clip
from pipeline_qc.image_qc_methods import (file_processing_methods, intensity,
                                          query_fovs, z_stack_check)


class StandardizeFOVArrayResult(NamedTuple):
    fov_id: int
    stat_dict: dict


class StandardizeFOVArrayError(NamedTuple):
    fov_id: int
    error: str

def process_single_fov(row, json_dir, output_dir, image_gen=False, env='stg'):

    try:

        print(f"Processing fovid:{str(row['fovid'])}")

        # Doesn't run the code if a picke for the fovid identified already exists
        if os.path.isfile(f"{json_dir}/{row['fovid']}.pickle"):
            print(f"Fovid:{str(row['fovid'])} has already been processed")
            with open(f"{json_dir}/{row['fovid']}.pickle", 'rb') as handle:
                return StandardizeFOVArrayResult(row['fovid'], pickle.load(handle))

        # Splits 6D image into single channel images for qc algorithm processing
        channel_dict = file_processing_methods.split_image_into_channels(row['localfilepath'],
                                                                         str(row['sourceimagefileid']))

        # Segments area in FOV that has cells
        cell_mask = detect_edge.segment_from_zstack(channel_dict['brightfield'], gaussian_thresh=0.045)

        # Initializes a dictionary where all stats for an fov are saved
        stat_dict = dict()

        # Iterates over each z-stack image and runs qc_algorithms, and then adds each stat generated to the stat_dict
        for channel_name, channel_array in channel_dict.items():
            if channel_array.shape[0] == 1:
                print('This FOV is not a multi-dimensional image, skipping...')
                return dict()
            # Runs the intensity metrics on all z-stack images. Put here since run on all channels
            intensity_dict = intensity.intensity_stats_single_channel(channel_array, cell_mask)
            for intensity_key, intensity_value, in intensity_dict.items():
                stat_dict.update({channel_name + ' ' + intensity_key + '-intensity': intensity_value})

            # Runs all metrics to be run on brightfield (edge detection, false clip bf) and makes bf qc_images
            if channel_name == 'brightfield':
                bf_edge_detect = detect_edge.detect_edge_position(channel_array)
                for edge_key, edge_value in bf_edge_detect.items():
                    stat_dict.update({channel_name + ' ' + edge_key: edge_value})
                bf_zstack_intensity = z_stack_check.z_stack_order_check(channel_array)
                for zstack_key, zstack_value in bf_zstack_intensity:
                    stat_dict.update({channel_name + ' ' + zstack_key: zstack_value})
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

        # TODO: Need to figure out how to make numpy array json serializable in dict (changed to pickle)
        with open(f"{json_dir}/{row['fovid']}.pickle", "wb") as write_out:
            pickle.dump(stat_dict, write_out, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metrics for fov in labkey
        file_processing_methods.insert_qc_data_labkey(row['fovid'], stat_dict, env)

        print(f"Finished processing fovid:{str(row['fovid'])}")

        return StandardizeFOVArrayResult(row['fovid'], stat_dict)

    except Exception as e:
        print(f"Failed processing for FOV:{row['fovid']}")
        return StandardizeFOVArrayError(row['fovid'], str(e))

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

    # Create or get log dir
    # Do not include ms
    log_dir_name = datetime.now().isoformat().split(".")[0]
    log_dir = Path(f".dask_logs/{log_dir_name}").expanduser()
    # Log dir settings
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure dask config
    dask.config.set(
        {
            "scheduler.work-stealing": False,
        }
    )

    # Create cluster
    cluster = SLURMCluster(
        cores=1,
        memory="24GB",
        queue="aics_cpu_general",
        walltime="10:00:00",
        local_directory=str(log_dir),
        log_directory=str(log_dir),
    )

    # Scale cluster
    cluster.scale(128)

    print(f"Dask dashboard available at: {cluster.dashboard_link}")

    # Map fov processing in parallel to cluster
    with DistributedHandler(cluster.scheduler_address) as handler:
        results = handler.batched_map(
            process_single_fov,
            [row for i, row in query_df.iterrows()],
            [json_dir for i in range(len(query_df))],
            [output_dir for i in range(len(query_df))],
            [image_gen for i in range(len(query_df))],
            [env for i in range(len(query_df))],
        )
    stat_list = []
    errors = []
    for result in results:
        if isinstance(result, StandardizeFOVArrayResult):
            stat_list.append(result.stat_dict)
        else:
            errors.append(result.fov_id, result.error)

    # Joins query_df to stat_list, and then writes out a csv of all the data to an output folder

    stat_df = pd.concat([query_df, pd.DataFrame(stat_list)], axis=1)
    stat_df.to_csv(output_dir + '/fov_qc_metrics.csv')
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(output_dir + '/errors.csv')

    return stat_df, errors
