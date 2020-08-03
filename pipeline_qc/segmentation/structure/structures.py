from dataclasses import dataclass

@dataclass
class StructureInfo:
    gene: str
    ml: bool # Indicate whether the structure is segmented using Machine Learning / Deep learning
    ml_model: str # Model to use for ML segmentations
    algorithm_name: str
    algorithm_version: str

class Structures:
    # no need for cardio structures in this list
    struct_map = {        
        "AAVS1": {"ml": True, "model": "structure_AAVS1_production", "algo_name": "ML AASV1 Structure Segmentation", "algo_version": "0.1.0"},
        "LMNB1": {"ml": True, "model": "LMNB1_morphological_production_alpha", "algo_name": "ML LMNB1 Structure Segmentation", "algo_version": "0.1.0"},
        "ACTB": {"ml": False, "algo_name": "Python ACTB structure segmentation", "algo_version": "1.2.0"},
        "ACTN1": {"ml": False, "algo_name": "Python ACTN1 structure segmentation", "algo_version": "1.1.3"},
        "ATP2A2": {"ml": False, "algo_name": "Python ATP2A2 structure segmentation", "algo_version": "1.0.0"},
        "CETN2": {"ml": False, "algo_name": "Python CETN2 structure segmentation", "algo_version": "1.1.0"},
        "CTNNB1": {"ml": False, "algo_name": "Python CTNNB1 structure segmentation", "algo_version": "1.1.0"},
        "DSP": {"ml": False, "algo_name": "Python DSP structure segmentation", "algo_version": "1.1.1"},
        "FBL": {"ml": False, "algo_name": "Python FBL structure segmentation", "algo_version": "1.1.3"},
        "GJA1": {"ml": False, "algo_name": "Python GJA1 structure segmentation", "algo_version": "1.1.0"},
        "HIST1H2BJ": {"alias": "H2B", "ml": False, "algo_name": "Python H2B structure segmentation", "algo_version": "1.0.0"},
        "LAMP1": {"ml": False, "algo_name": "Python LAMP1 structure segmentation", "algo_version": "1.1.0"},
        "LMNB1": {"ml": True, "model": "LMNB1_morphological_production_alpha", "algo_name": "ML LMNB1 Structure Segmentation", "algo_version": "0.1.0"},
        "MYH10": {"ml": False, "algo_name": "Python MYH10 structure segmentation", "algo_version": "1.2.0"},
        "NPM1": {"ml": False, "algo_name": "Python NPM1 structure segmentation", "algo_version": "1.1.0"},
        "NUP153": {"ml": False, "algo_name": "Python NUP153 structure segmentation", "algo_version": "1.0.0"},
        "PXN": {"ml": False, "algo_name": "Python PXN structure segmentation", "algo_version": "1.0.0"},
        "RAB5A": {"ml": False, "algo_name": "Python RAB5A structure segmentation", "algo_version": "1.0.0"},
        "SEC61B": {"ml": False, "algo_name": "Python SEC61B structure segmentation", "algo_version": "1.1.2"},
        "SLC25A17": {"ml": False, "algo_name": "Python SLC25A17 structure segmentation", "algo_version": "1.2.1"},
        "SMC1A": {"ml": False, "algo_name": "Python SMC1A structure segmentation", "algo_version": "1.0.0"},
        "SON": {"ml": False, "algo_name": "Python SON structure segmentation", "algo_version": "1.0.0"},
        "ST6GAL1": {"ml": False, "algo_name": "Python ST6GAL1 structure segmentation", "algo_version": "1.2.0"},
        "TJP1": {"ml": False, "algo_name": "Python TJP1 structure segmentation", "algo_version": "1.1.1"},
        "TOMM20": {"ml": False, "algo_name": "Python TOMM20 structure segmentation", "algo_version": "1.1.2"},
        "TUBA1B": {"ml": False, "algo_name": "Python TUBA1B structure segmentation", "algo_version": "1.1.2"}
    }

    @classmethod
    def get(cls, gene: str) -> StructureInfo:
        """
        Get structure information for the given gene
        """
        gene = gene.upper()

        if gene not in cls.struct_map:
            return None

        return StructureInfo(gene=cls.struct_map[gene].get("alias", gene), #some structs have a shorter alias for segmentation
                             ml=cls.struct_map[gene]["ml"],
                             ml_model=cls.struct_map[gene].get("model", None),
                             algorithm_name=cls.struct_map[gene]["algo_name"],
                             algorithm_version=cls.struct_map[gene]["algo_version"])

