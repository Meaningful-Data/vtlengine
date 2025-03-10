import json
from pathlib import Path

import pytest
from pysdmx.model import RulesetScheme, TransformationScheme, UserDefinedOperatorScheme

from vtlengine.API import create_ast, load_vtl
from vtlengine.API._InternalApi import ast_to_sdmx
from vtlengine.AST import TimeAggregation
from vtlengine.AST.ASTEncoders import ComplexDecoder, ComplexEncoder
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
    node = TimeAggregation(op="time_agg", period_to="A", period_from=None, operand=None, conf=None)

    with pytest.raises(SemanticError, match="1-1-19-11"):
        interpreter.visit_TimeAggregation(node)
