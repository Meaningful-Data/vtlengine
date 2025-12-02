import json
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest

from vtlengine.API import create_ast, load_vtl
from vtlengine.AST import (
    ID,
    Aggregation,
    Analytic,
    Argument,
    Assignment,
    BinOp,
    CaseObj,
    DefIdentifier,
    DPRule,
    DPRuleset,
    HRBinOp,
    HRule,
    HRuleset,
    HRUnOp,
    Identifier,
    JoinOp,
    Operator,
    OrderBy,
    PersistentAssignment,
    RegularAggregation,
    Start,
    TimeAggregation,
    VarID,
)
from vtlengine.AST.ASTEncoders import ComplexDecoder, ComplexEncoder
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.DataTypes import ScalarType
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer

base_path = Path(__file__).parent
filepath = base_path / "data" / "encode"

param = [
    "DS_r := DS_1 + DS_2;",
    """DS_r := DS_1 + DS_2;
         //comment""",
]


param_ast = [
    ("DS_r := DS_1 + 5; DS_r := DS_1 * 10;", "1-3-3"),
    ("DS_r := DS_1 + 5; DS_r <- DS_1 * 10;", "1-3-3"),
    ("DS_r <- DS_1 + 5; DS_r <- DS_1 * 10;", "1-3-3"),
]


@pytest.mark.parametrize("script", param)
def test_encode_ast(script):
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    result = json.dumps(ast, indent=4, cls=ComplexEncoder)
    with open(filepath / "reference_encode.json", "r") as file_reference:
        reference = file_reference.read()
    assert result == reference


@pytest.mark.parametrize("script", param)
def test_decode_ast(script):
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    with open(filepath / "reference_encode.json") as file:
        ast_decode = json.load(file, object_hook=ComplexDecoder.object_hook)
    assert ast_decode == ast


def test_visit_TimeAggregation_error():
    interpreter = InterpreterAnalyzer(datasets={})
    node = TimeAggregation(
        op="time_agg",
        period_to="A",
        period_from=None,
        operand=None,
        conf=None,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )

    with pytest.raises(SemanticError, match="1-1-19-11"):
        interpreter.visit_TimeAggregation(node)


def test_visit_Start():
    visitor = ASTTemplate()
    child1 = VarID(value="child1", line_start=1, column_start=1, line_stop=1, column_stop=1)
    child2 = VarID(value="child2", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node = Start(
        children=[child1, child2], line_start=1, column_start=1, line_stop=1, column_stop=1
    )
    with mock.patch.object(visitor, "visit_VarID") as mock_visit:
        visitor.visit_Start(node)
        mock_visit.assert_any_call(child1)
        mock_visit.assert_any_call(child2)
        mock_visit.assert_called_with(child2)
        assert mock_visit.call_count == 2


def test_visit_Assignment():
    visitor = ASTTemplate()
    node_left = VarID(value="left", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node_right = VarID(value="right", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node = Assignment(
        left=node_left,
        op=":=",
        right=node_right,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit_VarID") as mock_visit:
        visitor.visit_Assignment(node)
        mock_visit.assert_any_call(node_left)
        mock_visit.assert_any_call(node_right)
        mock_visit.assert_called_with(node_right)
        assert mock_visit.call_count == 2


def test_visit_PersistentAssignment():
    visitor = ASTTemplate()
    node_left = VarID(value="left", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node_right = VarID(value="right", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node = PersistentAssignment(
        left=node_left,
        op="<-",
        right=node_right,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit_VarID") as mock_visit:
        visitor.visit_PersistentAssignment(node)
        mock_visit.assert_any_call(node_left)
        mock_visit.assert_any_call(node_right)
        mock_visit.assert_called_with(node_right)
        assert mock_visit.call_count == 2


def test_visit_BinOp():
    visitor = ASTTemplate()
    node_left = VarID(value="left", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node_right = VarID(value="right", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node = BinOp(
        left=node_left,
        op="+",
        right=node_right,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit_VarID") as mock_visit:
        visitor.visit_BinOp(node)
        mock_visit.assert_any_call(node_left)
        mock_visit.assert_any_call(node_right)
        mock_visit.assert_called_with(node_right)
        assert mock_visit.call_count == 2


def test_visit_JoinOp():
    visitor = ASTTemplate()
    clause1 = VarID(value="clause1", line_start=1, column_start=1, line_stop=1, column_stop=1)
    clause2 = VarID(value="clause2", line_start=1, column_start=1, line_stop=1, column_stop=1)
    using_node = VarID(value="using", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node = JoinOp(
        op="JOIN",
        clauses=[clause1, clause2],
        using=using_node,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit_VarID") as mock_visit:
        visitor.visit_JoinOp(node)
        mock_visit.assert_any_call(clause1)
        mock_visit.assert_any_call(clause2)
        mock_visit.assert_called_with(using_node)
        assert mock_visit.call_count == 3


def test_visit_Identifier():
    visitor = ASTTemplate()
    node = Identifier(
        value="identifier", kind="string", line_start=1, column_start=1, line_stop=1, column_stop=1
    )
    result = visitor.visit_Identifier(node)
    assert result == "identifier"


def test_visit_ID():
    visitor = ASTTemplate()
    node = ID(value="ID", type_="string", line_start=1, column_start=1, line_stop=1, column_stop=1)
    result = visitor.visit_ID(node)
    assert result == "ID"


def test_visit_RegularAggregation():
    visitor = ASTTemplate()
    dataset_node = VarID(value="dataset", line_start=1, column_start=1, line_stop=1, column_stop=1)
    child_node1 = VarID(value="child1", line_start=1, column_start=1, line_stop=1, column_stop=1)
    child_node2 = VarID(value="child2", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node = RegularAggregation(
        dataset=dataset_node,
        op="Calc",
        children=[child_node1, child_node2],
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit_VarID") as mock_visit:
        visitor.visit_RegularAggregation(node)
        mock_visit.assert_any_call(dataset_node)
        mock_visit.assert_any_call(child_node1)
        mock_visit.assert_any_call(child_node2)
        assert mock_visit.call_count == 3


def test_visit_Aggregation():
    visitor = ASTTemplate()
    operand_node = VarID(value="Sum", line_start=1, column_start=1, line_stop=1, column_stop=1)
    group_node1 = VarID(value="group1", line_start=1, column_start=1, line_stop=1, column_stop=1)
    group_node2 = VarID(value="group2", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node = Aggregation(
        op="SUM",
        operand=operand_node,
        grouping_op="group by",
        grouping=[group_node1, group_node2],
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit_VarID") as mock_visit:
        visitor.visit_Aggregation(node)
        mock_visit.assert_any_call(operand_node)
        mock_visit.assert_any_call(group_node1)
        mock_visit.assert_any_call(group_node2)
        assert mock_visit.call_count == 3


def test_visit_Analytic():
    visitor = ASTTemplate()
    operand_node = VarID(value="operand", line_start=1, column_start=1, line_stop=1, column_stop=1)
    partition_by = ["component1", "component2"]
    order_by = [
        OrderBy(
            component="component1",
            order="asc",
            line_start=1,
            column_start=1,
            line_stop=1,
            column_stop=1,
        ),
        OrderBy(
            component="component2",
            order="desc",
            line_start=1,
            column_start=1,
            line_stop=1,
            column_stop=1,
        ),
    ]
    node = Analytic(
        op="SUM",
        operand=operand_node,
        partition_by=partition_by,
        order_by=order_by,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )

    with mock.patch.object(visitor, "visit", wraps=visitor.visit) as mock_visit:
        visitor.visit_Analytic(node)
        mock_visit.assert_any_call(operand_node)
        assert mock_visit.call_count == 1


def test_visit_CaseObj():
    visitor = ASTTemplate()
    condition_node = VarID(
        value="condition", line_start=1, column_start=1, line_stop=1, column_stop=1
    )
    then_op_node = VarID(value="thenOp", line_start=1, column_start=1, line_stop=1, column_stop=1)
    node = CaseObj(
        condition=condition_node,
        thenOp=then_op_node,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit_VarID") as mock_visit:
        visitor.visit_CaseObj(node)
        mock_visit.assert_any_call(condition_node)
        mock_visit.assert_any_call(then_op_node)
        assert mock_visit.call_count == 2


def test_visit_Operator():
    visitor = ASTTemplate()
    visitor.visit = Mock()
    param1 = Argument(
        name="param1",
        type_=ScalarType,
        default=None,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    param2 = Argument(
        name="param2",
        type_=ScalarType,
        default=None,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    expression_node = VarID(
        value="expression",
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    node = Operator(
        op="ADD",
        parameters=[param1, param2],
        output_type="integer",
        expression=expression_node,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    visitor.visit_Operator(node)
    visitor.visit.assert_any_call(param1)
    visitor.visit.assert_any_call(param2)
    visitor.visit.assert_any_call(expression_node)
    assert visitor.visit.call_count == 3


def test_visit_Argument():
    visitor = ASTTemplate()
    visitor.visit = Mock()
    node = Argument(
        name="arg1",
        type_=ScalarType,
        default=None,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    visitor.visit_Argument(node)


def test_visit_HRuleset():
    visitor = ASTTemplate()
    element_node = DefIdentifier(
        value="Identifier",
        kind="Identifier",
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    left_operand = DefIdentifier(
        value="Identifier",
        kind="Identifier",
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    right_operand = DefIdentifier(
        value="Identifier",
        kind="Identifier",
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    bin_op_node = HRBinOp(
        left=left_operand,
        op="+",
        right=right_operand,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    rule_node = HRule(
        name="rule1",
        rule=bin_op_node,
        erCode="er001",
        erLevel=1,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    node = HRuleset(
        name="ruleset1",
        signature_type="accounting_entry",
        element=element_node,
        rules=[rule_node],
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit", wraps=visitor.visit) as mock_visit:
        visitor.visit_HRuleset(node)
        mock_visit.assert_any_call(element_node)
        mock_visit.assert_any_call(rule_node)
        mock_visit.assert_any_call(bin_op_node)
        mock_visit.assert_any_call(left_operand)
        mock_visit.assert_any_call(right_operand)
        assert mock_visit.call_count == 5


def test_visit_DPRule():
    visitor = ASTTemplate()
    left_node = DefIdentifier(
        value="Identifier",
        kind="Identifier",
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    right_node = DefIdentifier(
        value="Identifier",
        kind="Identifier",
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    rule_node = HRBinOp(
        left=left_node,
        op="=",
        right=right_node,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    dp_rule_node = DPRule(
        name="dp_rule_name",
        rule=rule_node,
        erCode="4",
        erLevel=1,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit", wraps=visitor.visit) as mock_visit:
        visitor.visit_DPRule(dp_rule_node)
        mock_visit.assert_any_call(rule_node)
        assert mock_visit.call_count == 3


def test_visit_DPRuleset():
    visitor = ASTTemplate()
    element_node = DefIdentifier(
        value="element", kind="kind", line_start=1, column_start=1, line_stop=1, column_stop=1
    )
    rule_node1 = DPRule(
        name="dprule1",
        rule=HRBinOp(
            left=element_node,
            op="=",
            right=element_node,
            line_start=1,
            column_start=1,
            line_stop=1,
            column_stop=1,
        ),
        erCode="1",
        erLevel=1,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    rule_node2 = DPRule(
        name="dprule2",
        rule=HRBinOp(
            left=element_node,
            op=">",
            right=element_node,
            line_start=1,
            column_start=1,
            line_stop=1,
            column_stop=1,
        ),
        erCode="2",
        erLevel=2,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    dp_ruleset_node = DPRuleset(
        name="dp_ruleset_name",
        signature_type="accounting_entry",
        params=[element_node],
        rules=[rule_node1, rule_node2],
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    with mock.patch.object(visitor, "visit", wraps=visitor.visit) as mock_visit:
        visitor.visit_DPRuleset(dp_ruleset_node)
        mock_visit.assert_any_call(element_node)
        mock_visit.assert_any_call(rule_node1)
        mock_visit.assert_any_call(rule_node2)
        assert mock_visit.call_count == 9


def test_visit_HRUnOp():
    visitor = ASTTemplate()
    operand_node = DefIdentifier(
        value="operand", kind="Identifier", line_start=1, column_start=1, line_stop=1, column_stop=1
    )
    node = HRUnOp(
        op="-", operand=operand_node, line_start=1, column_start=1, line_stop=1, column_stop=1
    )
    with mock.patch.object(visitor, "visit", wraps=visitor.visit) as mock_visit:
        visitor.visit_HRUnOp(node)
        mock_visit.assert_any_call(operand_node)
        assert mock_visit.call_count == 1


def test_visit_DefIdentifier():
    visitor = ASTTemplate()
    node = Mock()
    node.value = "identifier_value"
    result = visitor.visit_DefIdentifier(node)
    assert result == "identifier_value"


def test_visit_DPRIdentifier():
    visitor = ASTTemplate()
    node = Mock()
    node.value = "dpr_identifier_value"
    result = visitor.visit_DPRIdentifier(node)
    assert result == "dpr_identifier_value"


@pytest.mark.parametrize("script, error", param_ast)
def test_error_DAG_two_outputs_same_name(script, error):
    with pytest.raises(SemanticError, match=error):
        create_ast(text=script)


def test_rule_name_not_in_ruleset():
    script = """
    DS_r := check_hierarchy(DS_1, rule_count);
    """
    with pytest.raises(SemanticError, match="1-4-2-8"):
        create_ast(text=script)
