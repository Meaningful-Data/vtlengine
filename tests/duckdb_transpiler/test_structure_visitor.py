"""Tests for StructureVisitor class."""

from typing import Any, Dict, List

from vtlengine.AST import VarID
from vtlengine.DataTypes import Number, String
from vtlengine.duckdb_transpiler.Transpiler.structure_visitor import StructureVisitor
from vtlengine.Model import Component, Dataset, Role


def make_ast_node(**kwargs: Any) -> Dict[str, Any]:
    """Create common AST node parameters."""
    return {"line_start": 1, "column_start": 1, "line_stop": 1, "column_stop": 10, **kwargs}


def create_simple_dataset(name: str, id_cols: List[str], measure_cols: List[str]) -> Dataset:
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


class TestStructureVisitorVarID:
    """Test VarID structure computation."""

    def test_visit_varid_returns_structure_from_available_tables(self):
        """Test that visiting a VarID returns structure from available_tables."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(
            available_tables={"DS_1": ds1},
            output_datasets={},
        )

        varid = VarID(**make_ast_node(value="DS_1"))
        result = visitor.visit(varid)

        assert result is not None
        assert result.name == "DS_1"
        assert "Id_1" in result.components
        assert "Me_1" in result.components

    def test_visit_varid_returns_structure_from_output_datasets(self):
        """Test that visiting a VarID returns structure from output_datasets."""
        ds_r = create_simple_dataset("DS_r", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(
            available_tables={},
            output_datasets={"DS_r": ds_r},
        )

        varid = VarID(**make_ast_node(value="DS_r"))
        result = visitor.visit(varid)

        assert result is not None
        assert result.name == "DS_r"

    def test_visit_varid_with_udo_param_resolves_binding(self):
        """Test that VarID resolves UDO parameter bindings."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(
            available_tables={"DS_1": ds1},
            output_datasets={},
        )
        # Simulate UDO call: define myop(ds) = ds + 1
        # When called as myop(DS_1), ds is bound to VarID("DS_1")
        ds_param = VarID(**make_ast_node(value="DS_1"))
        visitor.push_udo_params({"ds": ds_param})

        varid = VarID(**make_ast_node(value="ds"))
        result = visitor.visit(varid)

        assert result is not None
        assert result.name == "DS_1"

    def test_visit_varid_returns_none_for_unknown(self):
        """Test that visiting unknown VarID returns None."""
        visitor = StructureVisitor(available_tables={}, output_datasets={})

        varid = VarID(**make_ast_node(value="UNKNOWN"))
        result = visitor.visit(varid)

        assert result is None
