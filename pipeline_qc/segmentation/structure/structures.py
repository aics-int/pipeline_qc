from dataclasses import dataclass

@dataclass
class StructureInfo:
    gene: str
    ml: bool # Indicate whether the structure is segmented using Machine Learning / Deep learning
    ml_model: str # Model to use for ML segmentations
    algorithm_name: str
    algorithm_version: str

class Structures:
    struct_map = {
        "H2B": {"ml": True, "model": "structure_H2B_production", "algo": "ML H2B Structure Segmentation", "algo_version": "0.1.0"},
        "AASV1": {"ml": True, "model": "structure_AAVS1_production", "algo": "ML AASV1 Structure Segmentation", "algo_version": "0.1.0"},
        "LMNB1": {"ml": True, "model": "LMNB1_morphological_production_alpha", "algo": "ML LMNB1 Structure Segmentation", "algo_version": "0.1.0"},
        "ACTB": {"ml": False, "algo": "Python ACTB structure segmentation", "algo_version": "1.2.0"},
        "ACTN1": {"ml": False, "algo": "Python ACTN1 structure segmentation", "algo_version": "1.1.3"},
        "CETN2": {"ml": False, "algo": "Python CETN2 structure segmentation", "algo_version": "1.1.0"},
        "CTNNB1": {"ml": False, "algo": "Python CTNNB1 structure segmentation", "algo_version": "1.1.0"},
        "DSP": {"ml": False, "algo": "Python DSP structure segmentation", "algo_version": "1.1.1"},
        "FBL": {"ml": False, "algo": "Python FBL structure segmentation", "algo_version": "1.1.3"},
        "GJA1": {"ml": False, "algo": "Python GJA1 structure segmentation", "algo_version": "1.1.0"},
        "LAMP1": {"ml": False, "algo": "Python LAMP1 structure segmentation", "algo_version": "1.2.0"},
        "MYH10": {"ml": False, "algo": "Python MYH10 structure segmentation", "algo_version": "1.2.0"},
        "NPM1": {"ml": False, "algo": "Python NPM1 structure segmentation", "algo_version": "1.1.0"},
        "NUP153": {"ml": False, "algo": "Python NUP153 structure segmentation", "algo_version": "1.1.0"},
        "PXN": {"ml": False, "algo": "Python PXN structure segmentation", "algo_version": "1.0.0"},
        "RAB5A": {"ml": False, "algo": "Python RAB5A structure segmentation", "algo_version": "1.0.0"},
        "SEC61B": {"ml": False, "algo": "Python SEC61B structure segmentation", "algo_version": "1.1.2"},
        "SLC25A17": {"ml": False, "algo": "Python SLC25A17 structure segmentation", "algo_version": "1.2.1"},
        "SON": {"ml": False, "algo": "", "algo_version": ""}, #TODO 
        "ST6GAL1": {"ml": False, "algo": "Python ST6GAL1 structure segmentation", "algo_version": "1.2.0"},
        "TJP1": {"ml": False, "algo": "	Python TJP1 structure segmentation", "algo_version": "1.1.1"},
        "TOMM20": {"ml": False, "algo": "Python TOMM20 structure segmentation", "algo_version": "1.1.2"},
        "TUBA1B": {"ml": False, "algo": "Python TUBA1B structure segmentation", "algo_version": "1.1.2"}
    }

    @classmethod
    def get(cls, gene: str) -> StructureInfo:
        """
        Get structure information for the given gene
        """
        gene = gene.upper()

        if gene not in cls.struct_map:
            return None

        return CellStructure(gene=gene, 
                             ml=cls.struct_map[gene]["ml"],
                             ml_model=cls.struct_map[gene].get("model", None),
                             algorithm_name=cls.struct_map[gene]["algo_name"],
                             algorithm_version=cls.struct_map[gene]["algo_version"])
 