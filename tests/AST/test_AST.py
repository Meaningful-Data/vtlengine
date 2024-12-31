import json
from pathlib import Path

import pytest
from vtlengine.API import load_vtl, create_ast
from vtlengine.AST.ASTEncoders import ComplexEncoder, ComplexDecoder


base_path = Path(__file__).parent
filepath = base_path / "data" / "encode"

param = [
    "DS_r := DS_1 + DS_2;"
]


@pytest.mark.parametrize("script", param)
def test_encode_ast(script):
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    result = json.dumps(ast, indent=4, cls=ComplexEncoder)
    with open(filepath / "reference_encode.json", 'r') as file_reference:
        reference = file_reference.read()
    assert result == reference


@pytest.mark.parametrize("script", param)
def test_decode_ast(script):
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    with open(filepath / 'reference_encode.json') as file:
        ast_decode = json.load(file, object_hook=ComplexDecoder.object_hook)
    assert ast_decode == ast


