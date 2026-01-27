from dataclasses import dataclass
from typing import Dict, Tuple

from duckdb_transpiler.Model import Dataset, Scalar
from vtlengine import AST
from vtlengine.AST.ASTTemplate import ASTTemplate

SQL_OP_MAPPING: Dict[str, str] = {
    "mod": "%",
    "len": "LENGTH",
    "ucase": "UPPER",
    "lcase": "LOWER",
}


def get_sql_op(op: str) -> str:
    """Get the SQL equivalent of a given operator."""
    return SQL_OP_MAPPING.get(op, op.upper())


def _sql_literal(constant: AST.Constant) -> str:
    """Convert a Constant to SQL literal."""
    if constant is None or constant.value is None:
        return "NULL"

    type_name = constant.type_
    if type_name == "String":
        escaped = str(constant.value).replace("'", "''")
        return f"'{escaped}'"
    elif type_name == "Integer":
        return str(int(constant.value))
    elif type_name == "Number":
        return str(float(constant.value))
    elif type_name == "Boolean":
        return "TRUE" if constant.value else "FALSE"
    return str(constant.value)


@dataclass
class SQLTranspiler(ASTTemplate):
    input_datasets: Dict[str, Dataset]
    output_datasets: Dict[str, Dataset]
    input_scalars: Dict[str, Scalar]
    output_scalars: Dict[str, Scalar]

    def __post_init__(self):
        self.all_datasets = {**self.input_datasets, **self.output_datasets}
        self.all_scalars = {**self.input_scalars, **self.output_scalars}
        self.all_operands = {**self.all_datasets, **self.all_scalars}

    def transpile(self, ast: AST) -> Dict[str, str]:
        """Transpile the AST to SQL queries."""
        return self.visit(ast)

    def get_operand_alias(self) -> str:
        alias = f"op_{self.operand_counter}"
        self.operand_counter += 1
        return alias

    def visit_Start(self, node: AST.Start) -> Dict[str, str]:
        """Visit the start node of the AST."""
        queries = {}
        self.operand_counter = 0

        for child in node.children:
            result = self.visit(child)
            if result:
                name = child.left.value
                queries[name] = result
        return queries

    def visit_Assignment(self, node: AST.Assignment) -> Tuple[str, str]:
        """Visit an assignment node."""
        return self.visit(node.right)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Tuple[str, str]:
        """Visit a persistent assignment node."""
        return self.visit(node.right)

    def visit_VarID(self, node: AST.VarID) -> str:
        """Process a variable identifier."""
        return f'"{node.value}"'

    def visit_Constant(self, node: AST.Constant) -> str:
        """Process a literal value."""
        return _sql_literal(node)

    def visit_BinOp(self, node: AST.BinOp) -> str:
        """Process a binary node."""
        sql_op = get_sql_op(node.op)
        left_sql = self.visit(node.left)
        right_sql = self.visit(node.right)
        return f"({left_sql} {sql_op} {right_sql})"

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        """Process a unary node."""
        sql_op = get_sql_op(node.op)
        operand_sql = self.visit(node.operand)
        return f"({sql_op} {operand_sql})"

    def visit_ParamOp(self, node: AST.ParamOp) -> str:
        """Process a parameterized node."""
        sql_op = get_sql_op(node.op)
        params_sql = ", ".join(self.visit(param) for param in node.params)
        return f"{sql_op}({params_sql})"

    def visit_If(self, node: AST.If) -> str:
        """Process if-then-else node."""
        condition = self.visit(node.condition)
        then_op = self.visit(node.thenOp)
        else_op = self.visit(node.elseOp)
        return f"CASE WHEN {condition} THEN {then_op} ELSE {else_op} END"

    def visit_Case(self, node: AST.Case) -> str:
        """Process case node."""
        cases = [self.visit(case_obj) for case_obj in node.cases]
        cases_sql = " ".join(cases)
        else_op = self.visit(node.elseOp)
        return f"CASE {cases_sql} ELSE {else_op} END"

    def visit_CaseObj(self, node: AST.CaseObj) -> str:
        """Process a single case object."""
        cond = self.visit(node.condition)
        then = self.visit(node.thenOp)
        return f"WHEN {cond} THEN {then}"

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> str:
        """
        Process clause operations (calc, filter, keep, drop, rename, etc.).

        These operate on a single dataset and modify its structure or data.
        """
        ...
