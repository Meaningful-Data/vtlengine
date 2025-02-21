import json
from pathlib import Path

import pytest
from pysdmx.model import RulesetScheme, TransformationScheme, UserDefinedOperatorScheme

from vtlengine.API import ast_to_sdmx, create_ast, load_vtl
from vtlengine.AST.ASTEncoders import ComplexDecoder, ComplexEncoder

base_path = Path(__file__).parent
filepath = base_path / "data" / "encode"

param = ["DS_r := DS_1 + DS_2;"]

params_to_sdmx = [
    ("DS_r := DS_1 + DS_2;", "MD", "2.1"),
    ("DS_r <- DS_1 + 1;", "MD", "2.1"),
    (
        """
    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
        FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
        sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
        sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1
        end datapoint ruleset;
    """,
        "MD",
        "2.1",
    ),
    (
        """
        define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;
        """,
        "MD",
        "2.1",
    ),
    (
        """
        define operator suma (ds1 dataset, ds2 dataset)
            returns dataset is
            ds1 + ds2
        end operator;
    """,
        "MD",
        "2.1",
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
    assert result.id == f"TS{agency_id}"
    assert result.vtl_version == version
    assert isinstance(result.ruleset_schemes[0], RulesetScheme)
    assert result.ruleset_schemes[0].id == f"RS{agency_id}"
    assert result.ruleset_schemes[0].vtl_version == version
    assert isinstance(result.user_defined_operator_schemes[0], UserDefinedOperatorScheme)
    assert result.user_defined_operator_schemes[0].id == f"UDO{agency_id}"
    assert result.user_defined_operator_schemes[0].vtl_version == version
