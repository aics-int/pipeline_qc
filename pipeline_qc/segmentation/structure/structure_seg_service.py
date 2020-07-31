import aicsimageio
import logging
import numpy as np
import traceback

from typing import List
from pathlib import Path
from tempfile import TemporaryDirectory
from aicsimageio.writers import ome_tiff_writer
from pipeline_qc.image_qc_methods import file_processing_methods, query_fovs
from model_zoo_3d_segmentation.zoo import SuperModel
from .structure_seg_repository import StructureSegmentationRepository
from ..configuration import AppConfig
from .structures import Structures, StructureInfo
from ..common.fov_file import FovFile
from ..common.segmentation_result import SegmentationResult, ResultStatus
from aicssegmentation.structure_wrapper.structure_segmenter import StructureSegmenter


class StructureSegmentationService:
    """
    Structure Segmentation Service
    Exposes functionality to perform specific structure segmentations on FOVs
    """
    def __init__(self, legacy_structure_segmenter: StructureSegmenter, repository: StructureSegmentationRepository, config: AppConfig):
        if repository is None:
            raise AttributeError("repository")
        if config is None:
            raise AttributeError("app_config")
        self._repository = repository
        self._config = config
        self._legacy_structure_segmenter = legacy_structure_segmenter
        self.log = logging.getLogger(__name__)

    def get_fov_records(self, workflows: List, plates: List, cell_lines: List, fovids: List, only_from_fms:bool) -> List[FovFile]:
        """
        Query for FOV records and return results in list form
        """
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                         only_from_fms=only_from_fms, labkey_host=self._config.labkey_host, labkey_port=self._config.labkey_port)
        
        fovs = []
        for index, row in query_df.iterrows():
            fovs.append(FovFile.from_dataframe_row(row))

        return fovs

    def structure_segmentation(self, 
                               fov: FovFile, 
                               save_to_fms: bool, 
                               save_to_filesystem: bool, 
                               output_dir: str, 
                               process_duplicates: bool) -> SegmentationResult: 
        """
        Run segmentation process for a single FOV
        :param: fov: FOV record
        :param: save_to_fms: indicate whether to save segmentation output to FMS
        :param: save_to_filesystem: indicate whether to save segmentation output to output_dir
        :param: output_dir: output directory path when saving to file system (can be network / isilon path)
        :param: process_duplicates: indicate whether to process or skip fov if segmentation already exists in FMS
        """               

        fov_id = fov.fov_id
        local_file_path = fov.local_file_path
        source_file_id = fov.source_image_file_id
        
        try:
            structure = Structures.get(fov.gene)
            if structure is None:
                msg = f"FOV {fov_id}: unsupported structure: {fov.gene}"
                self.log.info(msg)
                return SegmentationResult(fov_id=fov_id, status=ResultStatus.FAILED, message=msg)

            if not process_duplicates and self._repository.segmentation_exists(fov_id, structure):
                msg = f"FOV {fov_id} has already been segmented"
                self.log.info(msg)
                return SegmentationResult(fov_id=fov_id, status=ResultStatus.SKIPPED, message=msg)

            im = self._create_segmentable_image(fov, structure)
            if im is None:
                msg = f"FOV {fov_id} incompatible: missing channels or dimensions"
                self.log.info(msg)
                return SegmentationResult(fov_id=fov_id, status=ResultStatus.FAILED, message=msg)

            # Segment
            self.log.info(f'Running structure segmentation on FOV {fov_id}')

            structure_segmentation, structure_contour = self._segment_image(im, structure)
            if structure_segmentation is None:
                msg = f"FOV {fov_id} could not be segmented: returned empty result"
                self.log.info(msg)
                return SegmentationResult(fov_id=fov_id, status=ResultStatus.FAILED, message=msg)

            # Handle outputs
            struct_file_name, contour_file_name = self._get_seg_filename(local_file_path)

            if save_to_fms:
                self.log.info("Uploading structure segmentation to FMS")

                with TemporaryDirectory() as tmp_dir:
                    # Segmentation
                    seg_file_path = f'{tmp_dir}/{struct_file_name}'
                    with ome_tiff_writer.OmeTiffWriter(seg_file_path) as writer:
                        writer.save(structure_segmentation)
                    
                    if structure_contour is not None:
                        # Contour                
                        contour_file_path = f'{tmp_dir}/{contour_file_name}'
                        with ome_tiff_writer.OmeTiffWriter(contour_file_path) as writer:
                            writer.save(structure_contour)
                        self._repository.upload_structure_segmentation(structure, source_file_id, seg_file_path, contour_file_path)                       
                    else:     
                        self._repository.upload_structure_segmentation(structure, source_file_id, seg_file_path)


            if save_to_filesystem:
                self.log.info("Saving structure segmentation to filesystem")
                # Segmentation
                with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{struct_file_name}') as writer:
                    writer.save(structure_segmentation)
                # Contour
                if structure_contour is not None:
                    with ome_tiff_writer.OmeTiffWriter(f'{output_dir}/{contour_file_name}') as writer:
                        writer.save(structure_contour)                    

            return SegmentationResult(fov_id=fov_id, status=ResultStatus.SUCCESS)

        except Exception as ex:
            msg = f"Exception while processing FOV {fov_id}: {str(ex)}\n{traceback.format_exc()}"
            self.log.info(msg)
            return SegmentationResult(fov_id=fov_id, status=ResultStatus.FAILED, message=msg)

    def _segment_image(self, image: np.array, structure_info: StructureInfo):
        """
        Perform structure segmentation
        return: (structure_segmentation, structure_contour)
        """
        if structure_info.ml:
            result = self._segment_from_model(image, structure_info.ml_model)
        else:
            result = self._segment_from_legacy_wrapper(image, structure_info)
        
        # format return so we can unpack into 2 values (seg, contour) no matter what gets returned by the segmentation
        if(type(result) == tuple and len(result) >= 2):
            return (result[0], result[1])
        
        return (result, None)
        
    def _segment_from_model(self, image: np.array, model):
        """
        Segment using ML model
        Uses core ML segmentation code from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/dl_model_zoo/browse 
        """
        sm = SuperModel(model)

        return sm.apply_on_single_zstack(input_img=image)

    def _segment_from_legacy_wrapper(self, image: np.array, structure_info: StructureInfo) -> (np.array, np.array):
        """
        Segment using legacy wrappers from https://aicsbitbucket.corp.alleninstitute.org/projects/ASSAY/repos/aics-segmentation/browse 
        return: (structure_segmentation, structure_contour)
        """
        gene = structure_info.gene
        return self._legacy_structure_segmenter.process_img(gene, image)


    def _create_segmentable_image(self, fov: FovFile, structure_info: StructureInfo) -> np.array:
        """
        Create a segmentable image
        param: fov: fov file info
        param: structure_info: structure info
        """
        if structure_info.ml:
            return self._create_segmentable_image_ml(fov)
        else:
            return self._create_segmentable_image_legacy(fov)

    def _create_segmentable_image_ml(self, fov: FovFile) -> np.array:
        """
        Create a segmentable image by picking the right channels
        ML structure segmentation requires an image with 2 channels: 488nm (structure) and 638nm (membrane)
        """
        aicsimageio.use_dask(False) # disable dask image reads to avoid losing performance when running on GPU nodes

        channel_dict = file_processing_methods.split_image_into_channels(fov.local_file_path, fov.source_image_file_id)

        full_im_list = list()

        for channel in ['488nm', '638nm']:
            for channel_name, channel_array in channel_dict.items():
                if channel_array.shape[0] <= 1: 
                    return None # not a 3D image
                if channel_name == channel:
                    full_im_list.append(channel_array)
        
        if(len(full_im_list) != 2):
            return None

        return np.array(full_im_list)

    def _create_segmentable_image_legacy(self, fov: FovFile) -> np.array:
        """
        Create a segmentable image by picking the right channels
        Legacy structure segmentation requires the image structure channel only
        """
        aicsimageio.use_dask(False) # disable dask image reads to avoid losing performance when running on GPU nodes

        channel_dict = file_processing_methods.split_image_into_channels(fov.local_file_path, fov.source_image_file_id)
        
        if "488nm" not in channel_dict:
            return None

        struct_channel = channel_dict["488nm"]
        if struct_channel.shape[0] <= 1:
            return None # not a 3D image
        
        return struct_channel

    def _get_seg_filename(self, fov_file_path: str) -> (str, str):
        """
        Generate appropriate segmentation filenames based on FOV file name
        Will look like this: 
            structure: {barcode}-{obj}-{date}-{colony pos(optional)}-{scene}-{pos}-{well}_struct_segmentation.tiff
            contour: {barcode}-{obj}-{date}-{colony pos(optional)}-{scene}-{pos}-{well}_struct_contour.tiff
        param: fov_file_path: source fov file path
        return: (struct_seg_file_name, struct_contour_file_name)
        """
        if fov_file_path.endswith(".ome.tiff"):
            file_prefix = Path(fov_file_path[:-9]).stem
        else:
            file_prefix = Path(fov_file_path).stem
        
        file_prefix = file_prefix.replace("-alignV2", "").replace("alignV2", "") # get rid of alignV2 in all its forms
        
        return (f"{file_prefix}_struct_segmentation.tiff", f"{file_prefix}_struct_contour.tiff")
        
