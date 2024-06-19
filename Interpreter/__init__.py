import AST
from AST.ASTTemplate import ASTTemplate


class InterpreterAnalyzer(ASTTemplate):

    def visit_Start(self, node: AST.Start) -> None:
        for child in node.children:
            self.visit(child)
        # TODO: Execute collected operations from Spark

    def visit_Assignment(self, node: AST.Assignment) -> None:
        left_operand = self.visit(node.left)
        right_operand = self.visit(node.right)
        # TODO Assignment evaluate

    def visit_BinOp(self, node: AST.BinOp) -> None:
        left_operand = self.visit(node.left)
        right_operand = self.visit(node.right)
        # TODO BinOp evaluate

    def visit_UnaryOp(self, node: AST.UnaryOp) -> None:
        operand = self.visit(node.operand)
        # TODO UnaryOp evaluate

    def visit_VarID(self, node: AST.VarID) -> AST.AST:
        # TODO VarID evaluate
        pass

    def visit_Constant(self, node: AST.Constant) -> AST.AST:
        # TODO Constant evaluate
        pass
