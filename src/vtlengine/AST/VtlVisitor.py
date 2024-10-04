from pathlib import Path

from antlr4 import ParseTreeVisitor

from vtlengine.AST.Grammar.parser import VtlParser


# This class defines a complete generic visitor for a parse tree produced by Parser.

class VtlVisitor(ParseTreeVisitor):
    pass

attr = []
for attr_name in dir(VtlParser):
    if attr_name.endswith("Context") and attr_name not in attr and attr_name != "RuleContext" and attr_name[0].isupper():
        attr.append(attr_name)
        method_name = f"visit{attr_name}"
        method_code = f"# Visit a parse tree produced by VtlParser#{attr_name}.\n"
        method_code += f"def {method_name}(self, ctx: VtlParser.{attr_name}):\n    return self.visitChildren(ctx)"
        file_path = Path(__file__)
        with open(file_path, "a") as f:
            f.write(method_code + "\n\n")

if __name__ == '__main__':
    visitor = VtlVisitor()