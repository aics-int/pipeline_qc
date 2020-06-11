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

class CellSegmentationDistributedWrapper2:
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

    def _create_segmentable_image(self, row):
        print(f"** START create_segmentable_image: {datetime.now()}")
        localfilepath = row['localfilepath']
        sourceimagefileid = row['sourceimagefileid']

        channel_dict = file_processing_methods.split_image_into_channels(localfilepath, sourceimagefileid)

        full_im_list = list()
        for channel in ['405nm', '638nm', 'brightfield']:
            for key, value in channel_dict.items():
                if key == channel:
                    full_im_list.append(value)

        print(f"** END create_segmentable_image: {datetime.now()}")
        return np.array(full_im_list)

    def batch_cell_segmentations(self, workflows=None, cell_lines=None, plates=None, fovids=None,
                                 only_from_fms=True, save_to_fms=False, save_to_isilon=False,
                                 output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'):
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                        only_from_fms=only_from_fms, labkey_context=self._labkey_context)
        print(f'''
        __________________________________________

        {len(query_df)} fovs were found to process.

        __________________________________________
        ''')

        # Create segmentable images (CPU)
        print(f"START CPU work: {datetime.now()}")
        
        rows = []
        for i, row in query_df.iterrows():
            rows.append(row)
        
        cluster = SLURMCluster(cores=1, 
                               memory="50G", 
                               queue="aics_cpu_general",
                               nanny=True,
                               walltime="00:30:00")     
        cluster.scale(20)
        print(cluster.job_script())

        images = []
        with DistributedHandler(cluster.scheduler_address) as handler:
            futures = handler.client.map(
                lambda row: self._create_segmentable_image(row),
                rows
            )

            images = handler.gather(futures)

        print(f"END CPU work: {datetime.now()}")      


        # Segment (GPU)
        print(f"START GPU work: {datetime.now()}")
        cluster = SLURMCluster(cores=1, 
                               memory="60G", 
                               queue="aics_gpu_general",
                               nanny=False,
                               walltime="00:30:00",
                               extra=["--resources GPU=1"],
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

        print(f"END GPU work: {datetime.now()}")

    def _process_single_cell_segmentation(self, im, row, output_dir, save_to_fms, save_to_isilon):
        print(f"** START DISTRIBUTED::cell_seg_wrapper_distributed._process_single_cell_segmentation: {datetime.now()}")
        fov_id = row["fovid"]

        try:
            file_name = self._get_seg_filename(row['localfilepath'])
                                            
            print(f"** START single_seg_run: {datetime.now()}")
            comb_seg = self.single_seg_run(im)
            print(f"** END single_seg_run: {datetime.now()}")
            
            return f"FOV {fov_id} success" 

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



