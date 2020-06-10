import numpy as np
import os
import traceback

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from datetime import datetime
from aicsimageio.writers import ome_tiff_writer
from pipeline_qc.image_qc_methods import file_processing_methods, query_fovs
from model_zoo_3d_segmentation.zoo import SegModel, SuperModel
from pipeline_qc.image_qc_methods.cell_seg_uploader import CellSegmentationUploader
from labkey.utils import ServerContext
from aics_dask_utils import DistributedHandler
from dask_jobqueue import SLURMCluster

# Constants
MODEL = "DNA_MEM_instance_LF_integration_two_camera"

class CellSegmentationDistributedWrapper:
    """
    Single cell ML Segmentation wrapper
    Wraps the core segmentation code from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse
    and performs additional query and upload tasks for microscopy pipeline usage
    """

    def __init__(self, uploader: CellSegmentationUploader, labkey_context: ServerContext):
        if not uploader:
            raise AttributeError("uploader")
        self._uploader = uploader
        self._labkey_context = labkey_context

    def single_seg_run(self, image):
        sm = SuperModel(MODEL)

        return sm.apply_on_single_zstack(input_img=image, inputCh=[0, 1, 2])

    def _create_segmentable_image(self, localfilepath, sourceimagefileid):

        channel_dict = file_processing_methods.split_image_into_channels(localfilepath, sourceimagefileid)

        full_im_list = list()
        for channel in ['405nm', '638nm', 'brightfield']:
            for key, value in channel_dict.items():
                if key == channel:
                    full_im_list.append(value)

        return np.array(full_im_list)

    def batch_cell_segmentations(self, workflows=None, cell_lines=None, plates=None, fovids=None,
                                 only_from_fms=True, save_to_fms=False, save_to_isilon=False,
                                 output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'):
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                        only_from_fms=only_from_fms, labkey_context=self._labkey_context)
        rows = []
        for i, row in query_df.iterrows():
            rows.append(row)

        print(f'''
        __________________________________________

        {len(query_df)} fovs were found to process.

        __________________________________________
        ''')

        cluster = SLURMCluster(cores=1, 
                               memory="60G", 
                               queue="aics_gpu_general",
                               nanny=False,
                               walltime="00:30:00",
                               extra=["--resources GPU=1,nthreads4"],
                               job_extra=["--gres=gpu:gtx1080:1"])   
        cluster.scale(3)
        print(cluster.job_script())

        with DistributedHandler(cluster.scheduler_address) as handler:
            futures = handler.client.map(
                lambda row: self._process_single_cell_segmentation(row, output_dir, save_to_fms, save_to_isilon),
                rows
            )

            results = handler.gather(futures)
            print("Results:\n")
            for r in results:
                print(f"{r}\n")



    def _process_single_cell_segmentation(self, row, output_dir, save_to_fms, save_to_isilon):
        fov_id = row["fovid"]

        try:
            file_name = self._get_seg_filename(row['localfilepath'])
                
            if os.path.isfile(f'{output_dir}/{file_name}'):
                msg = f'FOV:{row["fovid"]} has already been segmented'
                print(msg)
                return msg
            else:
                print(f'Running Segmentation on fov:{row["fovid"]}')
                im = self._create_segmentable_image(row['localfilepath'], row['sourceimagefileid'])
                if im.shape[0] ==3:
                    comb_seg = self.single_seg_run(im)
                else:
                    msg = f'FOV:{row["fovid"]} does not have nucleus or cellular color channels'
                    print(msg)
                    return msg
                
                if save_to_fms == True:
                    print("Uploading output file to FMS")

                    with TemporaryDirectory() as tmp_dir:
                        local_file_path = f'{tmp_dir}/{file_name}'
                        with ome_tiff_writer.OmeTiffWriter(local_file_path) as writer:
                            writer.save(comb_seg)
                        self._uploader.upload_combined_segmentation(local_file_path, row["sourceimagefileid"])

                if save_to_isilon == True:
                    print("Saving output file to Isilon")
                    with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{file_name}') as writer:
                        writer.save(comb_seg)

        except Exception as e:
            error = f"FOV {fov_id} failure: {str(e)}\n{traceback.format_exc()}"
            print(error)
            return error 

    def _get_seg_filename(self, fov_file_path: str):
        """
        Generate appropriate segmentation filename based on FOV file name
        Will look like this: {barcode}-{obj}-{date}-{colony pos(optional)}-{scene}-{pos}-{well}_CellNucSegCombined.ome.tiff
        """
        if fov_file_path.endswith(".ome.tiff"):
            file_prefix = Path(fov_file_path[:-9]).stem
        else:
            file_prefix = Path(fov_file_path).stem
        
        file_prefix = file_prefix.replace("-alignV2", "").replace("alignV2", "") # get rid of alignV2 in all its forms
        return f"{file_prefix}_CellNucSegCombined.ome.tiff"



