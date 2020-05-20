import numpy as np
import os

from typing import Dict
from datetime import datetime
from aicsfiles import FileManagementSystem
from aicsfiles.filter import Filter
from aicsimageio.writers import ome_tiff_writer
from hpctools.interfaces import ContentTypes, LabkeyInterface
from pipeline_qc.image_qc_methods import file_processing_methods, query_fovs
from model_zoo_3d_segmentation.zoo import SegModel, SuperModel


# Constants
ALGORITHM = "dna_cell_segmentation_ML_v1"
ALGORITHM_VERSION = "0.1.0"
MODEL = "DNA_MEM_instance_LF_integration_two_camera"


class CellSegmentationWrapper:
    """
    Single cell ML Segmentation wrapper
    Wraps the core segmentation code from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse
    and performs additional query and upload tasks for microscopy pipeline usage
    """

    def __init__(self, uploader: CellSegmentationUploader):
        self._uploader = uploader or CellSegmentationUploader()

    def single_seg_run(self, image):
        sm = SuperModel(MODEL)

        return sm.apply_on_single_zstack(input_img=image, inputCh=[0, 1, 2])

    def create_segmentable_image(self, localfilepath, sourceimagefileid):

        channel_dict = file_processing_methods.split_image_into_channels(localfilepath, sourceimagefileid)

        full_im_list = list()
        for channel in ['405nm', '638nm', 'brightfield']:
            for key, value in channel_dict.items():
                if key == channel:
                    full_im_list.append(value)

        return np.array(full_im_list)

    def batch_cell_segmentations(self, workflows=None, cell_lines=None, plates=None, fovids=None,
                                 only_from_fms=True, save_to_fms=False, save_to_isilon=True,
                                 output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'):
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                        only_from_fms=only_from_fms)

        print(f'''
        __________________________________________

        {len(query_df)} fovs were found to process.

        __________________________________________
        ''')

        for index, row in query_df.iterrows():

            if os.path.isfile(f'{output_dir}/{row["fovid"]}.ome.tif'):
                print(f'FOV:{row["fovid"]} has already been segmented')
            else:
                print(f'Running Segmentation on fov:{row["fovid"]}')
                im = create_segmentable_image(row['localfilepath'], row['sourceimagefileid'])
                comb_seg = single_seg_run(im)

                if save_to_fms == True:
                    print("Upload output file to FMS")  #TODO: move outside of loop and batch upload?

                if save_to_isilon == True:
                    print("Saving output file to Isilon")
                    with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{row["fovid"]}.ome.tif') as writer:
                        writer.save(comb_seg)

        return


class CellSegmentationUploader:
    """
    Interface for uploading segmentation files to FMS/Labkey
    """
    def __init__(self, fms_config: Dict , fms_client: FileManagementSystem = None): #TODO timeout
        self._fms_timeout = int(fms_config.get("fms_timeout_in_seconds", 300))
        self._fms_client = fms_client or FileManagementSystem(host=fms_config.get("fms_host", "stg-aics"),
                                                              port=fms_config.get("fms_port", 80))

    def upload_combined_segmentation(self, combined_segmentation_path: str, input_file_name: str):
        """
        Augment with proper metadata and upload a combined segmentation files to FMS
        :param: combined_segmentation_path: combined segmentation output file path
        :param: input_file_name: file name of the input image used to produce this segmentation (used to gather metadata)
        """

        # Minimal Metadata structure:
        #
        # "file": {
        #     "file_type": "image"
        # },
        # "content_processing": {
        #     "channels": {
        #         "0": {
        #             "algorithm": <algorithm name, as recorded in Labkey>,
        #             "algorithm_version": <algorithm version, as recorded in Labkey>,
        #             "content_type": <ContentType, as recorded in Labkey>,
        #             "processing_date": <processing date>,
        #         },
        #         "1": {
        #           ...
        #         },
        #         ...
        #     }
        # }

        # Channel 0 = Nucleus segmentation
        # Channel 1 = Membrane segmentation
        # Channel 2 = Nucleus contour
        # Channel 3 = Membrane contour

        processing_date: str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

        # Initialize metadata from input file's metadata, or start over if not available
        metadata = self._get_file_metadata(input_file_name) or {}
        metadata.update({"file": {"file_type": "image"}})

        metadata["content_processing"] = {
            "channels": {
                "0": self._channel_metadata_block(ContentTypes.NucSeg, processing_date),
                "1": self._channel_metadata_block(ContentTypes.MembSeg, processing_date),
                "2": self._channel_metadata_block(ContentTypes.NucContour, processing_date),
                "3": self._channel_metadata_block(ContentTypes.MembContour, processing_date)
            }
        }

        self._fms_client.upload_file(combined_segmentation_path, metadata, timeout=self._fms_timeout)

    def _channel_metadata_block(self, content_type: str, processing_date: str):
        """
        Build and return a metadata block for a given channel
        param: content_type: content type to identify the channel's contents
        param: processing_date: content processing date
        """
        return {
                "algorithm": ALGORITHM,
                "algorithm_version": ALGORITHM_VERSION,
                "content_type": content_type,
                "processing_date": processing_date
                }

    def _get_file_metadata(self, file_name: str):
        """
        Get file metadata from FMS
        param: file_name: name of the file in FMS
        return: file metadata if the file was found, None otherwise
        """
        query = Filter().with_file_name(file_name)
        result = self._fms_client.query_files(query)

        return result[0] if result and len(result) > 0 else None
