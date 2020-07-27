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
        "AASV1":    {"ml": True, "model": "structure_AAVS1_production", "algo": "ML AASV1 Structure Segmentation", "algo_version": "0.1.0"},
        "ACTB":     {"ml": False, "algo": "Python ACTB structure segmentation", "algo_version": "1.2.0",'module': 'aicssegmentation.structure_wrapper.seg_actb', 'class': 'ACTB_HiPSC_Pipeline'},
        "ACTN1":    {"ml": False, "algo": "Python ACTN1 structure segmentation", "algo_version": "1.1.3",'module': 'aicssegmentation.structure_wrapper.seg_actn1', 'class': 'ACTN1_HiPSC_Pipeline'},
        "CETN2":    {"ml": False, "algo": "Python CETN2 structure segmentation", "algo_version": "1.1.0",'module': 'aicssegmentation.structure_wrapper.seg_cetn2', 'class': 'CETN2_HiPSC_Pipeline'},
        "CTNNB1":   {"ml": False, "algo": "Python CTNNB1 structure segmentation", "algo_version": "1.1.0",'module': 'aicssegmentation.structure_wrapper.seg_ctnnb1', 'class': 'CTNNB1_HiPSC_Pipeline'},
        "DSP":      {"ml": False, "algo": "Python DSP structure segmentation", "algo_version": "1.1.1",'module': 'aicssegmentation.structure_wrapper.seg_dsp', 'class': 'DSP_HiPSC_Pipeline'},
        "FBL":      {"ml": False, "algo": "Python FBL structure segmentation", "algo_version": "1.1.3",'module': 'aicssegmentation.structure_wrapper.seg_fbl', 'class': 'FBL_HiPSC_Pipeline'},
        "GJA1":     {"ml": False, "algo": "Python GJA1 structure segmentation", "algo_version": "1.1.0",'module': 'aicssegmentation.structure_wrapper.seg_gja1', 'class': 'GJA1_HiPSC_Pipeline'},
        "H2B":      {"ml": True, "model": "structure_H2B_production", "algo": "ML H2B Structure Segmentation", "algo_version": "0.1.0"},
        "LAMP1":    {"ml": False, "algo": "Python LAMP1 structure segmentation", "algo_version": "1.1.0",'module': 'aicssegmentation.structure_wrapper.seg_lamp1', 'class': 'LAMP1_HiPSC_Pipeline'},
        "LMNB1":    {"ml": True, "model": "LMNB1_morphological_production_alpha", "algo": "ML LMNB1 Structure Segmentation", "algo_version": "0.1.0"},
        "MYH10":    {"ml": False, "algo": "Python MYH10 structure segmentation", "algo_version": "1.2.0",'module': 'aicssegmentation.structure_wrapper.seg_myh10', 'class': 'MYH10_HiPSC_Pipeline'},
        "NPM1":     {"ml": False, "algo": "Python NPM1 structure segmentation", "algo_version": "1.1.0",'module': 'aicssegmentation.structure_wrapper.seg_npm1', 'class': 'NPM1_HiPSC_Pipeline'},
        "NUP153":   {"ml": False, "algo": "Python NUP153 structure segmentation", "algo_version": "1.1.0", 'module':'aicssegmentation.structure_wrapper.seg_nup153', 'class':'NUP153_HiPSC_Pipeline'},
        "PXN":      {"ml": False, "algo": "Python PXN structure segmentation", "algo_version": "1.0.0",'module': 'aicssegmentation.structure_wrapper.seg_pxn', 'class': 'PXN_HiPSC_Pipeline'},
        "RAB5A":    {"ml": False, "algo": "Python RAB5A structure segmentation", "algo_version": "1.0.0",'module': 'aicssegmentation.structure_wrapper.seg_rab5a', 'class': 'RAB5A_HiPSC_Pipeline'},
        "SEC61B":   {"ml": False, "algo": "Python SEC61B structure segmentation", "algo_version": "1.1.2",'module': 'aicssegmentation.structure_wrapper.seg_sec61b', 'class': 'SEC61B_HiPSC_Pipeline'},
        "SLC25A17": {"ml": False, "algo": "Python SLC25A17 structure segmentation", "algo_version": "1.2.1",'module': 'aicssegmentation.structure_wrapper.seg_slc25a17', 'class': 'SLC25A17_HiPSC_Pipeline'},
        "SON":      {"ml": False, "algo": "Python SON structure segmentation", "algo_version": "1.0.0", 'module':'aicssegmentation.structure_wrapper.seg_son', 'class':'SON_HiPSC_Pipeline'}, #TODO verify
        "ST6GAL1":  {"ml": False, "algo": "Python ST6GAL1 structure segmentation", "algo_version": "1.2.0",'module': 'aicssegmentation.structure_wrapper.seg_st6gal1', 'class': 'ST6GAL1_HiPSC_Pipeline'},
        "TJP1":     {"ml": False, "algo": "	Python TJP1 structure segmentation", "algo_version": "1.1.1",'module': 'aicssegmentation.structure_wrapper.seg_tjp1', 'class': 'TJP1_HiPSC_Pipeline'},
        "TOMM20":   {"ml": False, "algo": "Python TOMM20 structure segmentation", "algo_version": "1.1.2",'module': 'aicssegmentation.structure_wrapper.seg_tomm20', 'class': 'TOMM20_HiPSC_Pipeline'},
        "TUBA1B":   {"ml": False, "algo": "Python TUBA1B structure segmentation", "algo_version": "1.1.2",'module': 'aicssegmentation.structure_wrapper.seg_tuba1b', 'class': 'TUBA1B_HiPSC_Pipeline'}
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

