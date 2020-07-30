import pytest

from pipeline_qc.segmentation.structure.structures import Structures, StructureInfo

class TestStructures:
    def test_structures_get_valid_structure(self):
        # Act
        struct: StructureInfo = Structures.get("H2B")
        
        # Assert
        assert struct is not None
        assert struct.gene == "H2B"
        assert struct.ml == True
        assert struct.ml_model == "structure_H2B_production"
        assert struct.algorithm_name == "ML H2B Structure Segmentation"
        assert struct.algorithm_version == "0.1.0"

    def test_structures_get_invalid_structure(self):
        # Assert
        assert Structures.get("NOT_A_STRUCTURE") is None
