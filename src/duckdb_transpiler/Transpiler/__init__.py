from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from duckdb_transpiler.Utils import get_sql_op, sql_literal
from vtlengine import AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.Model import Dataset, Scalar


@dataclass
class SQLTranspiler(ASTTemplate):
    """Transpiler from VTL AST to SQL queries using unified JOIN optimization."""

    datasets: Dict[str, Dataset] = field(default_factory=dict)
    scalars: Dict[str, Scalar] = field(default_factory=dict)

    # Current query context
    _current_query: Any = field(default=None, init=False)

    def transpile(self, ast: AST.Start, queries: List[Any]) -> List[Any]:
        """Transpile the AST to SQL queries."""
        self._queries = queries
        return self.visit(ast)

    # =========================================================================
    # Visitor Methods
    # =========================================================================

    def visit_Start(self, node: AST.Start) -> List[Any]:
        """Visit start node - process each assignment."""
        for child, query in zip(node.children, self._queries):
            self._current_query = query
            query.sql = self.visit(child)
            # Register output for subsequent queries
            if isinstance(query.structure, Dataset):
                self.datasets[query.name] = query.structure
            else:
                self.scalars[query.name] = query.structure
        return self._queries

    def visit_Assignment(self, node: AST.Assignment) -> str:
        """Visit assignment node."""
        return self._generate_query(node.right)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> str:
        """Visit persistent assignment node."""
        return self._generate_query(node.right)

    def visit_VarID(self, node: AST.VarID) -> str:
        """Visit variable identifier."""
        name = node.value
        if name in self.scalars:
            scalar = self.scalars[name]
            if scalar.value is not None:
                return sql_literal(scalar.value, scalar.data_type.__name__)
            # Scalar without value - reference as subquery from registered table
            return f'(SELECT value FROM "{name}")'
        return f'"{name}"'

    def visit_Constant(self, node: AST.Constant) -> str:
        """Visit constant literal."""
        return sql_literal(node.value, node.type_)

    def visit_BinOp(self, node: AST.BinOp) -> str:
        """Visit binary operation - scalar context only."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {get_sql_op(node.op)} {right})"

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        """Visit unary operation - scalar context only."""
        operand = self.visit(node.operand)
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)
        sql_op = get_sql_op(op)

        if op in ("+", "-"):
            return f"({sql_op}{operand})"
        if op == "not":
            return f"(NOT {operand})"
        if op == "isnull":
            return f"({operand} IS NULL)"
        return f"{sql_op}({operand})"

    def visit_ParamOp(self, node: AST.ParamOp) -> str:
        """Visit parameterized operation - scalar context only."""
        if not node.children:
            return ""
        operand = self.visit(node.children[0])
        params = [self.visit(p) for p in node.params] if node.params else []
        sql_op = get_sql_op(node.op.lower() if isinstance(node.op, str) else str(node.op))
        return f"{sql_op}({', '.join([operand] + params)})"

    def visit_ParFunction(self, node: AST.ParFunction) -> str:
        """Visit parenthesized expression."""
        return self.visit(node.operand)

    # =========================================================================
    # Query Generation
    # =========================================================================

    def _generate_query(self, node: AST.AST) -> str:
        """Generate SQL query for an expression."""
        structure = self._current_query.structure

        # Scalar output
        if isinstance(structure, Scalar):
            return f"SELECT {self.visit(node)} AS value"

        # Dataset output - use unified JOIN optimization
        return self._generate_dataset_query(node, structure)

    def _generate_dataset_query(self, node: AST.AST, structure: Dataset) -> str:
        """Generate optimized SQL for dataset expression using unified JOIN."""
        # Collect all base datasets referenced
        base_datasets = self._collect_datasets(node)

        if not base_datasets:
            # Pure scalar expression applied to structure
            return self.visit(node)

        # Build unified FROM clause with JOINs
        from_sql, alias_map = self._build_from_clause(base_datasets)

        # Build SELECT clause
        first_alias = alias_map[sorted(base_datasets)[0]]
        id_cols = [f'{first_alias}."{id_}"' for id_ in structure.get_identifiers_names()]
        measure_exprs = [
            f'{self._build_expression(node, alias_map, m)} AS "{m}"'
            for m in structure.get_measures_names()
        ]

        select = ", ".join(id_cols + measure_exprs)
        return f"SELECT {select}\nFROM {from_sql}"

    def _collect_datasets(self, node: AST.AST) -> Set[str]:
        """Collect all dataset names referenced in an expression."""
        if isinstance(node, AST.VarID):
            return {node.value} if node.value in self.datasets else set()
        if isinstance(node, AST.Constant):
            return set()
        if isinstance(node, AST.BinOp):
            return self._collect_datasets(node.left) | self._collect_datasets(node.right)
        if isinstance(node, AST.UnaryOp):
            return self._collect_datasets(node.operand)
        if isinstance(node, AST.ParFunction):
            return self._collect_datasets(node.operand)
        if isinstance(node, AST.ParamOp) and node.children:
            return self._collect_datasets(node.children[0])
        return set()

    def _build_from_clause(self, dataset_names: Set[str]) -> tuple[str, Dict[str, str]]:
        """Build FROM clause with JOINs for multiple datasets."""
        names = sorted(dataset_names)
        alias_map = {name: f"op_{i}" for i, name in enumerate(names)}

        if len(names) == 1:
            name = names[0]
            return f'"{name}" AS {alias_map[name]}', alias_map

        # Find common identifiers for USING clause
        datasets = [self.datasets[n] for n in names]
        common_ids = set(datasets[0].get_identifiers_names())
        for ds in datasets[1:]:
            common_ids &= set(ds.get_identifiers_names())

        if not common_ids:
            raise ValueError("No common identifiers across datasets")

        using = ", ".join(f'"{id_}"' for id_ in sorted(common_ids))

        # Build chained JOINs
        parts = [f'"{names[0]}" AS {alias_map[names[0]]}']
        for name in names[1:]:
            parts.append(f'INNER JOIN "{name}" AS {alias_map[name]} USING ({using})')

        return "\n".join(parts), alias_map

    def _build_expression(self, node: AST.AST, alias_map: Dict[str, str], measure: str) -> str:
        """Build SQL expression for a measure in the unified JOIN context."""
        if isinstance(node, AST.VarID):
            name = node.value
            if name in alias_map:
                return f'{alias_map[name]}."{measure}"'
            if name in self.scalars:
                scalar = self.scalars[name]
                if scalar.value is not None:
                    return sql_literal(scalar.value, scalar.data_type.__name__)
                # Scalar without value - reference as subquery
                return f'(SELECT value FROM "{name}")'
            return f'"{name}"'

        if isinstance(node, AST.Constant):
            return sql_literal(node.value, node.type_)

        if isinstance(node, AST.BinOp):
            left = self._build_expression(node.left, alias_map, measure)
            right = self._build_expression(node.right, alias_map, measure)
            return f"({left} {get_sql_op(node.op)} {right})"

        if isinstance(node, AST.UnaryOp):
            operand = self._build_expression(node.operand, alias_map, measure)
            op = node.op.lower() if isinstance(node.op, str) else str(node.op)
            sql_op = get_sql_op(op)
            if op in ("+", "-"):
                return f"({sql_op}{operand})"
            if op == "not":
                return f"(NOT {operand})"
            if op == "isnull":
                return f"({operand} IS NULL)"
            return f"{sql_op}({operand})"

        if isinstance(node, AST.ParFunction):
            return self._build_expression(node.operand, alias_map, measure)

        if isinstance(node, AST.ParamOp) and node.children:
            operand = self._build_expression(node.children[0], alias_map, measure)
            params = [self.visit(p) for p in node.params] if node.params else []
            sql_op = get_sql_op(node.op.lower() if isinstance(node.op, str) else str(node.op))
            return f"{sql_op}({', '.join([operand] + params)})"

        return "NULL"
