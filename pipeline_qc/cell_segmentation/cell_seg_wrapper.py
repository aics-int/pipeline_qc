import logging

from abc import ABC, abstractmethod
from aics_dask_utils import DistributedHandler
from dask_jobqueue import SLURMCluster
from pipeline_qc.image_qc_methods import query_fovs
from .cell_seg_service import CellSegmentationService
from .configuration import AppConfig, GpuClusterConfig

class CellSegmentationWrapperBase(ABC):
    """
    ML Single Cell Segmentation wrapper base
    """
    @abstractmethod
    def batch_cell_segmentations(self, workflows, cell_lines, plates, fovids,
                                 only_from_fms, save_to_fms, save_to_filesystem, process_duplicates, output_dir): 
        pass

class CellSegmentationWrapper(CellSegmentationWrapperBase):
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
        self.log = logging.getLogger(__name__)

    def batch_cell_segmentations(self, workflows=None, cell_lines=None, plates=None, fovids=None,
                                 only_from_fms=True, save_to_fms=False, save_to_filesystem=False, process_duplicates=False,
                                 output_dir = './output'): 
        """
        Process segmentations as a batch. 
        FOV images are queried from FMS based on the given options and segmented sequentially.
        """                                
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                        only_from_fms=only_from_fms, labkey_host=self._config.labkey_host, labkey_port=self._config.labkey_port)



        self.log.info(f"{len(query_df)} fovs were found to process.")


        for index, row in query_df.iterrows():
            self._cell_seg_service.single_cell_segmentation(row, 
                                                            save_to_fms=save_to_fms, 
                                                            save_to_filesystem=save_to_filesystem,
                                                            output_dir=output_dir,
                                                            process_duplicates=process_duplicates)


class CellSegmentationDistributedWrapper(CellSegmentationWrapperBase):
    """
    Distributed ML Single Cell Segmentation wrapper
    This wrapper uses Dask to distribute segmentation runs accross GPU nodes on the Slurm cluster 
    """

    def __init__(self, cell_seg_service: CellSegmentationService, config: AppConfig, cluster_config: GpuClusterConfig):
        if cell_seg_service is None:
            raise AttributeError("cell_seg_service")
        if config is None:
            raise AttributeError("app_config")
        if cluster_config is None:
            raise AttributeError("cluster_config")
        self._cell_seg_service = cell_seg_service
        self._config = config
        self._cluster_config = cluster_config
        self.log = logging.getLogger(__name__)

    def batch_cell_segmentations(self, workflows=None, cell_lines=None, plates=None, fovids=None,
                                 only_from_fms=True, save_to_fms=False, save_to_filesystem=False, process_duplicates=False,
                                 output_dir = './output'): 
        """
        Process segmentations as a batch. 
        FOV images are queried from FMS based on the given options and segmented concurrently on GPU nodes
        """                                
        query_df = query_fovs.query_fovs(workflows=workflows, plates=plates, cell_lines=cell_lines, fovids=fovids,
                                        only_from_fms=only_from_fms, labkey_host=self._config.labkey_host, labkey_port=self._config.labkey_port)


        self.log.info(f"{len(query_df)} fovs were found to process.")

        cluster = SLURMCluster(cores=1, 
                               memory=self._cluster_config.worker_memory_limit, 
                               queue=self._cluster_config.partition,
                               nanny=False,
                               walltime=self._cluster_config.worker_time_limit,
                               extra=["--resources GPU=1"],
                               job_extra=[f"--gres=gpu:{self._cluster_config.gpu}:1"])

        cluster.scale(self._cluster_config.cluster_size)

        self.log.debug(cluster.job_script())

        with DistributedHandler(cluster.scheduler_address) as handler:
            results = handler.batched_map(
                lambda row: self._cell_seg_service.single_cell_segmentation(row, 
                                                                            save_to_fms=save_to_fms, 
                                                                            save_to_filesystem=save_to_filesystem,
                                                                            output_dir=output_dir,
                                                                            process_duplicates=process_duplicates),
                [row for i, row in query_df.iterrows()],
                resources={"GPU":1}
            )

            self.log.info("Results:")
            for r in results:
                self.log.info(r)
        
