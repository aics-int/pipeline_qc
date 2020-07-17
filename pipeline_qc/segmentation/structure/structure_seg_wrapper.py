class StructureSegmentationWrapperBase(ABC):
    """
    Structure Segmentation wrapper base
    """
    @abstractmethod
    def batch_structure_segmentations(self, workflows, cell_lines, plates, fovids, #how do we pick structure??
                                      only_from_fms, save_to_fms, save_to_filesystem, process_duplicates, output_dir): 
        pass


class StructureSegmentationWrapper(StructureSegmentationWrapperBase):
    """
    Single process Structure Segmentation wrapper
    """
    pass


class StructureSegmentationWrapperDistributed(StructureSegmentationWrapperBase):
    """
    Distributed Structure Segmentation wrapper
    This wrapper uses Dask to distribute segmentation runs accross GPU nodes on the Slurm cluster 
    """

    # query
    #   ?? how do I distributed to CPU or GPU based on structure type? 
    #   for now -> run everything on GPU
    pass
