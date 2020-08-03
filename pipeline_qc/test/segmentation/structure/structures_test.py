import pytest

from pipeline_qc.segmentation.structure.structures import Structures, StructureInfo

class TestStructures:
    def test_structures_get_valid_structure(self):
        # Act
        struct: StructureInfo = Structures.get("AAVS1")
        
        # Assert
        assert struct is not None
        assert struct.gene == "AAVS1"
        assert struct.ml == True
        assert struct.ml_model == "structure_AAVS1_production"
        assert struct.algorithm_name == "ML AASV1 Structure Segmentation"
        assert struct.algorithm_version == "0.1.0"

    def test_structures_get_valid_structure_with_alias(self):
        # Act
        struct: StructureInfo = Structures.get("HIST1H2BJ")
        
        # Assert
        assert struct is not None
        assert struct.gene == "H2B"
        assert struct.ml == False
        assert struct.algorithm_name == "Python H2B structure segmentation"
        assert struct.algorithm_version == "1.0.0"

    def test_structures_get_invalid_structure(self):
        # Assert
        assert Structures.get("NOT_A_STRUCTURE") is None
