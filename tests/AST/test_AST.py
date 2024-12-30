import json

from vtlengine.API import load_vtl, create_ast
from vtlengine.AST.ASTEncoders import ComplexEncoder, ComplexDecoder

# AST generation
vtl = load_vtl(script)
ast = create_ast(vtl)
print(ast)

# Encode to JSON
result = json.dumps(ast, indent=4, cls=ComplexEncoder)
print(result)

# Decode from JSON
ast = json.loads(result, object_hook=ComplexDecoder.object_hook)
print(ast)