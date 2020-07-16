class StructureSegmentationWrapperBase(ABC):
    """
    Structure Segmentation wrapper base
    """
    @abstractmethod
    def batch_structure_segmentations(self, workflows, cell_lines, plates, fovids, #how do we pick structure??
                                      only_from_fms, save_to_fms, save_to_filesystem, process_duplicates, output_dir): 
        pass


class StructureSegmentationWrapper(StructureSegmentationWrapperBase):
    pass


class StructureSegmentationWrapperDistributed(StructureSegmentationWrapperBase):
    # query
    #   ?? how do I distributed to CPU or GPU based on structure type? 
    #   for now -> run everything on GPU
    pass
