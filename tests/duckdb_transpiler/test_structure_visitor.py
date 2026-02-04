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


class TestStructureVisitorUDOParams:
    """Test UDO parameter handling in StructureVisitor."""

    def test_get_udo_param_returns_none_when_no_params(self):
        """Test get_udo_param returns None when no UDO params are set."""
        visitor = StructureVisitor(available_tables={}, output_datasets={})
        assert visitor.get_udo_param("param1") is None

    def test_get_udo_param_finds_param_in_current_scope(self):
        """Test get_udo_param finds parameter in current scope."""
        visitor = StructureVisitor(available_tables={}, output_datasets={})
        visitor.push_udo_params({"param1": "value1"})

        assert visitor.get_udo_param("param1") == "value1"
        assert visitor.get_udo_param("nonexistent") is None

    def test_get_udo_param_searches_outer_scopes(self):
        """Test get_udo_param searches outer scopes for nested UDOs."""
        visitor = StructureVisitor(available_tables={}, output_datasets={})
        visitor.push_udo_params({"outer_param": "outer_value"})
        visitor.push_udo_params({"inner_param": "inner_value"})

        # Should find both inner and outer params
        assert visitor.get_udo_param("inner_param") == "inner_value"
        assert visitor.get_udo_param("outer_param") == "outer_value"

    def test_push_pop_udo_params_manages_stack(self):
        """Test push/pop correctly manages the UDO param stack."""
        visitor = StructureVisitor(available_tables={}, output_datasets={})

        visitor.push_udo_params({"a": 1})
        visitor.push_udo_params({"b": 2})

        assert visitor.get_udo_param("b") == 2

        visitor.pop_udo_params()

        assert visitor.get_udo_param("b") is None
        assert visitor.get_udo_param("a") == 1

        visitor.pop_udo_params()

        assert visitor.get_udo_param("a") is None
