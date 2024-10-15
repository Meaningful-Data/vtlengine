"""
AST.ASTVisitor.py
=================

Description
-----------
Node Dispatcher.
"""

from typing import Any


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
        raise Exception("No visit_{} method".format(type(node).__name__))
