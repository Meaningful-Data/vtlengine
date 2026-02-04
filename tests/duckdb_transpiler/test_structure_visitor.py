"""Tests for StructureVisitor class."""


from vtlengine.DataTypes import Number, String
from vtlengine.duckdb_transpiler.Transpiler.structure_visitor import StructureVisitor
from vtlengine.Model import Component, Dataset, Role


def create_simple_dataset(name: str, id_cols: list, measure_cols: list) -> Dataset:
    """Helper to create a simple Dataset for testing."""
    components = {}
    for col in id_cols:
        components[col] = Component(
            name=col, data_type=String, role=Role.IDENTIFIER, nullable=False
        )
    for col in measure_cols:
        components[col] = Component(name=col, data_type=Number, role=Role.MEASURE, nullable=True)
    return Dataset(name=name, components=components, data=None)


class TestStructureVisitorBasics:
    """Test basic StructureVisitor functionality."""

    def test_visitor_can_be_instantiated(self):
        """Test that StructureVisitor can be created."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(
            available_tables={"DS_1": ds1},
            output_datasets={},
        )
        assert visitor is not None

    def test_visitor_clear_context_resets_structure_cache(self):
        """Test that clear_context removes cached structures."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(
            available_tables={"DS_1": ds1},
            output_datasets={},
        )
        # Manually add something to context
        visitor._structure_context[123] = ds1
        assert len(visitor._structure_context) == 1

        visitor.clear_context()

        assert len(visitor._structure_context) == 0
