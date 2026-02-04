"""Tests for StructureVisitor class."""

from typing import Any, Dict, List

from vtlengine.AST import BinOp, Identifier, ParamOp, RegularAggregation, RenameNode, UnaryOp, VarID
from vtlengine.AST.Grammar.tokens import MEMBERSHIP
from vtlengine.DataTypes import Boolean, Integer, Number, String
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


class TestStructureVisitorBinOp:
    """Test BinOp structure computation."""

    def test_visit_binop_membership_extracts_single_measure(self):
        """Test that membership (#) returns structure with only extracted component."""
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
                "Me_2": Component(name="Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        membership = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="DS_1")),
                op=MEMBERSHIP,
                right=VarID(**make_ast_node(value="Me_1")),
            )
        )

        result = visitor.visit(membership)

        assert result is not None
        assert "Id_1" in result.components
        assert "Me_1" in result.components
        assert "Me_2" not in result.components
        assert result.components["Me_1"].role == Role.MEASURE

    def test_visit_binop_alias_returns_operand_structure(self):
        """Test that alias (as) returns same structure as operand."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        alias = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="DS_1")),
                op="as",
                right=Identifier(**make_ast_node(value="A", kind="DatasetID")),
            )
        )

        result = visitor.visit(alias)

        assert result is not None
        assert "Id_1" in result.components
        assert "Me_1" in result.components

    def test_visit_binop_arithmetic_returns_left_structure(self):
        """Test that arithmetic BinOp returns left operand structure."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        binop = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="DS_1")),
                op="+",
                right=VarID(**make_ast_node(value="DS_1")),
            )
        )

        result = visitor.visit(binop)

        assert result is not None
        assert "Id_1" in result.components
        assert "Me_1" in result.components


class TestStructureVisitorUnaryOp:
    """Test UnaryOp structure computation."""

    def test_visit_unaryop_isnull_returns_bool_var(self):
        """Test that isnull returns structure with bool_var measure."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        isnull = UnaryOp(
            **make_ast_node(
                op="isnull",
                operand=VarID(**make_ast_node(value="DS_1")),
            )
        )

        result = visitor.visit(isnull)

        assert result is not None
        assert "Id_1" in result.components
        assert "bool_var" in result.components
        assert "Me_1" not in result.components
        assert result.components["bool_var"].data_type == Boolean

    def test_visit_unaryop_other_returns_operand_structure(self):
        """Test that other unary ops return operand structure unchanged."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        abs_op = UnaryOp(
            **make_ast_node(
                op="abs",
                operand=VarID(**make_ast_node(value="DS_1")),
            )
        )

        result = visitor.visit(abs_op)

        assert result is not None
        assert "Id_1" in result.components
        assert "Me_1" in result.components


class TestStructureVisitorParamOp:
    """Test ParamOp structure computation."""

    def test_visit_paramop_cast_updates_measure_types(self):
        """Test that cast updates measure data types."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        cast_op = ParamOp(
            **make_ast_node(
                op="cast",
                children=[
                    VarID(**make_ast_node(value="DS_1")),
                    Identifier(**make_ast_node(value="Integer", kind="ScalarTypeConstraint")),
                ],
                params=[],
            )
        )

        result = visitor.visit(cast_op)

        assert result is not None
        assert "Id_1" in result.components
        assert "Me_1" in result.components
        assert result.components["Me_1"].data_type == Integer


class TestStructureVisitorRegularAggregation:
    """Test RegularAggregation (clause) structure computation."""

    def test_visit_keep_filters_components(self):
        """Test that keep clause removes unlisted components."""
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
                "Me_2": Component(name="Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        keep = RegularAggregation(
            **make_ast_node(
                op="keep",
                dataset=VarID(**make_ast_node(value="DS_1")),
                children=[VarID(**make_ast_node(value="Me_1"))],
            )
        )

        result = visitor.visit(keep)

        assert result is not None
        assert "Id_1" in result.components  # Identifiers always kept
        assert "Me_1" in result.components
        assert "Me_2" not in result.components

    def test_visit_drop_removes_components(self):
        """Test that drop clause removes listed components."""
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
                "Me_2": Component(name="Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        drop = RegularAggregation(
            **make_ast_node(
                op="drop",
                dataset=VarID(**make_ast_node(value="DS_1")),
                children=[VarID(**make_ast_node(value="Me_2"))],
            )
        )

        result = visitor.visit(drop)

        assert result is not None
        assert "Id_1" in result.components
        assert "Me_1" in result.components
        assert "Me_2" not in result.components

    def test_visit_rename_changes_component_names(self):
        """Test that rename clause changes component names."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        rename = RegularAggregation(
            **make_ast_node(
                op="rename",
                dataset=VarID(**make_ast_node(value="DS_1")),
                children=[RenameNode(**make_ast_node(old_name="Me_1", new_name="Me_1A"))],
            )
        )

        result = visitor.visit(rename)

        assert result is not None
        assert "Id_1" in result.components
        assert "Me_1" not in result.components
        assert "Me_1A" in result.components

    def test_visit_filter_preserves_structure(self):
        """Test that filter clause preserves structure."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        visitor = StructureVisitor(available_tables={"DS_1": ds}, output_datasets={})

        filter_op = RegularAggregation(
            **make_ast_node(
                op="filter",
                dataset=VarID(**make_ast_node(value="DS_1")),
                children=[
                    BinOp(
                        **make_ast_node(
                            left=VarID(**make_ast_node(value="Me_1")),
                            op=">",
                            right=VarID(**make_ast_node(value="0")),
                        )
                    )
                ],
            )
        )

        result = visitor.visit(filter_op)

        assert result is not None
        assert "Id_1" in result.components
        assert "Me_1" in result.components
