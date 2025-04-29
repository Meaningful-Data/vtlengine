import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pysdmx.model import RulesetScheme, TransformationScheme, UserDefinedOperatorScheme

from vtlengine.API import create_ast, load_vtl
from vtlengine.API._InternalApi import ast_to_sdmx
from vtlengine.AST import (
    ID,
    Aggregation,
    Analytic,
    Argument,
    Assignment,
    BinOp,
    CaseObj,
    DPRule,
    DPRuleset,
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

param = ["DS_r := DS_1 + DS_2;"]

params_to_sdmx = [
    ("DS_r := DS_1 + DS_2;", "MD", "1.0"),
    ("DS_r <- DS_1 + 1;", "MD", "1.0"),
    (
        """
    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
        FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
        sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
        sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1
        end datapoint ruleset;
    """,
        "MD",
        "1.0",
    ),
    (
        """
        define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;
        """,
        "MD",
        "1.0",
    ),
    (
        """
        define operator suma (ds1 dataset, ds2 dataset)
            returns dataset is
            ds1 + ds2
        end operator;
    """,
        "MD",
        "1.0",
    ),
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


@pytest.mark.parametrize("script, agency_id, version", params_to_sdmx)
def test_ast_to_sdmx(script, agency_id, version):
    ast = create_ast(script)
    result = ast_to_sdmx(ast, agency_id, version)
    assert isinstance(result, TransformationScheme)
    assert result.agency == agency_id
    assert result.id == "TS1"
    assert result.version == version
    assert result.vtl_version == "2.1"
    assert isinstance(result.ruleset_schemes[0], RulesetScheme)
    assert result.ruleset_schemes[0].id == "RS1"
    assert result.ruleset_schemes[0].agency == agency_id
    assert result.ruleset_schemes[0].version == version
    assert result.ruleset_schemes[0].vtl_version == "2.1"
    assert isinstance(result.user_defined_operator_schemes[0], UserDefinedOperatorScheme)
    assert result.user_defined_operator_schemes[0].id == "UDS1"
    assert result.user_defined_operator_schemes[0].agency == agency_id
    assert result.user_defined_operator_schemes[0].version == version
    assert result.user_defined_operator_schemes[0].vtl_version == "2.1"


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
    visitor.visit_Start(node)
    assert True


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
    visitor.visit_Assignment(node)
    assert True


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
    visitor.visit_PersistentAssignment(node)
    assert True


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
    visitor.visit_BinOp(node)
    assert True


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
    visitor.visit_JoinOp(node)
    assert True


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
    result = visitor.visit_RegularAggregation(node)
    assert result is None


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
    result = visitor.visit_Aggregation(node)
    assert result is None


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
    result = visitor.visit_Analytic(node)
    assert result is None


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
    result = visitor.visit_CaseObj(node)
    assert result is None


def test_visit_Operator():
    visitor = ASTTemplate()
    visitor.visit = Mock()
    param1 = Argument(
        name="param1",
        type_=Mock(spec=ScalarType),
        default=None,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    param2 = Argument(
        name="param2",
        type_=Mock(spec=ScalarType),
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
    result = visitor.visit_Operator(node)
    assert result is None
    visitor.visit.assert_any_call(param1)
    visitor.visit.assert_any_call(param2)
    visitor.visit.assert_any_call(expression_node)


def test_visit_Argument():
    from unittest.mock import Mock

    visitor = ASTTemplate()
    visitor.visit = Mock()
    type_node = Mock()
    default_node = Mock()
    node = Argument(
        name="arg1",
        type_=type_node,
        default=default_node,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    visitor.visit_Argument(node)
    visitor.visit.assert_any_call(type_node)
    visitor.visit.assert_any_call(default_node)


def test_visit_HRuleset():
    visitor = ASTTemplate()
    visitor.visit = Mock()
    element_node = Mock()
    rule1 = Mock()
    rule2 = Mock()
    node = HRuleset(
        name="ruleset1",
        signature_type="type1",
        element=element_node,
        rules=[rule1, rule2],
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    visitor.visit_HRuleset(node)
    visitor.visit.assert_any_call(element_node)
    visitor.visit.assert_any_call(rule1)
    visitor.visit.assert_any_call(rule2)


def test_visit_DPRuleset():
    visitor = ASTTemplate()
    visitor.visit = Mock()
    param1 = Mock()
    param2 = Mock()
    rule1 = Mock()
    rule2 = Mock()
    node = DPRuleset(
        name="DpRuleset1",
        params=[param1, param2],
        rules=[rule1, rule2],
        signature_type="variable",
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    visitor.visit_DPRuleset(node)
    visitor.visit.assert_any_call(param1)
    visitor.visit.assert_any_call(param2)
    visitor.visit.assert_any_call(rule1)
    visitor.visit.assert_any_call(rule2)


def test_visit_DPRule():
    visitor = ASTTemplate()
    visitor.visit = Mock()
    rule_node = Mock()
    er_code_node = Mock()
    er_level_node = Mock()
    node = DPRule(
        name="rule1",
        rule=rule_node,
        erCode=er_code_node,
        erLevel=er_level_node,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    visitor.visit_DPRule(node)
    visitor.visit.assert_any_call(rule_node)


def test_visit_HRUnOp():
    visitor = ASTTemplate()
    visitor.visit = Mock()
    operand_node = Mock()
    node = HRUnOp(
        op="+",
        operand=operand_node,
        line_start=1,
        column_start=1,
        line_stop=1,
        column_stop=1,
    )
    visitor.visit_HRUnOp(node)
    visitor.visit.assert_any_call(operand_node)


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
