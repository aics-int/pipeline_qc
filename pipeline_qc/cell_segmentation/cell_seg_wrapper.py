from pipeline_qc.image_qc_methods import query_fovs
from .cell_seg_service import CellSegmentationService
from .configuration import AppConfig

class CellSegmentationWrapper:
    """
    Single process ML Single Cell Segmentation wrapper
    """

    def __init__(self, cell_seg_service: CellSegmentationService, config: AppConfig):
        if cell_seg_service is None:
            raise AttributeError("cell_seg_service")
        if config is None:
            raise AttributeError("app_config")
        self._cell_seg_service = cell_seg_service
        self._config = config


    def batch_cell_segmentations(self, workflows=None, cell_lines=None, plates=None, fovids=None,
                                 only_from_fms=True, save_to_fms=False, save_to_filesystem=False, process_duplicates=False,
                                 output_dir = '/allen/aics/microscopy/Aditya/cell_segmentations'): 
        """
        Process segmentations as a batch. 
        FOV images are queried from FMS based on the given options and segmented sequentially.
        """                                
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                        only_from_fms=only_from_fms)


        print(f'''
        __________________________________________

        {len(query_df)} fovs were found to process.

        __________________________________________
        ''')

        for index, row in query_df.iterrows():
            self._cell_seg_service.single_cell_segmentation(row, 
                                                            save_to_fms=save_to_fms, 
                                                            save_to_filesystem=save_to_filesystem,
                                                            output_dir=output_dir,
                                                            process_duplicates=process_duplicates)
