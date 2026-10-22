"""
AST.ASTVisitor.py
=================

Description
-----------
Node Dispatcher.
"""

from typing import Any

from vtlengine.Exceptions import SemanticError


class NodeVisitor(object):
    """ """

    def visit(self, node: Any):
        """ """
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """ """
        # AST_ASTVISITOR.1
        raise SemanticError("1-3-14", location=f"NodeVisitor.generic_visit ({type(node).__name__})")
