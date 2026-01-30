from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from duckdb_transpiler.Utils import get_sql_op, sql_literal
from vtlengine import AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.Model import Dataset, Scalar


@dataclass
class ClauseContext:
    """Context for building SQL with clause operations (filter, calc, keep, drop, rename)."""

    base_source: str  # Base dataset name or subquery
    alias: str  # Alias for the base source
    column_mapping: Dict[str, str]  # Maps VTL component name -> SQL column reference
    where_conditions: List[str] = field(default_factory=list)
    calc_expressions: Dict[str, str] = field(default_factory=dict)  # new_name -> expression
    keep_columns: Optional[Set[str]] = None
    drop_columns: Set[str] = field(default_factory=set)
    rename_mapping: Dict[str, str] = field(default_factory=dict)  # old_name -> new_name


@dataclass
class SQLTranspiler(ASTTemplate):
    """Transpiler from VTL AST to SQL queries using unified JOIN optimization."""

    datasets: Dict[str, Dataset] = field(default_factory=dict)
    scalars: Dict[str, Scalar] = field(default_factory=dict)

    # Current query context
    _current_query: Any = field(default=None, init=False)
    _clause_context: Optional[ClauseContext] = field(default=None, init=False)

    def transpile(self, ast: AST.Start, queries: List[Any]) -> List[Any]:
        """Transpile the AST to SQL queries."""
        self._queries = queries
        return self.visit(ast)

    # =========================================================================
    # Main Visitor Methods
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

    # =========================================================================
    # Scalar Context Visitors (for simple expressions)
    # =========================================================================

    def visit_VarID(self, node: AST.VarID) -> str:
        """Visit variable identifier."""
        name = node.value
        # Check if in clause context (component reference)
        if self._clause_context and name in self._clause_context.column_mapping:
            return self._clause_context.column_mapping[name]
        if name in self.scalars:
            scalar = self.scalars[name]
            if scalar.value is not None:
                return sql_literal(scalar.value, scalar.data_type.__name__)
            return f'(SELECT value FROM "{name}")'
        return f'"{name}"'

    def visit_Constant(self, node: AST.Constant) -> str:
        """Visit constant literal."""
        return sql_literal(node.value, node.type_)

    def visit_BinOp(self, node: AST.BinOp) -> str:
        """Visit binary operation."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {get_sql_op(node.op)} {right})"

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        """Visit unary operation."""
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
        """Visit parameterized operation."""
        if not node.children:
            return ""
        operand = self.visit(node.children[0])
        params = [self.visit(p) for p in node.params] if node.params else []
        sql_op = get_sql_op(node.op.lower() if isinstance(node.op, str) else str(node.op))
        return f"{sql_op}({', '.join([operand] + params)})"

    def visit_ParFunction(self, node: AST.ParFunction) -> str:
        """Visit parenthesized expression."""
        return self.visit(node.operand)

    def visit_Identifier(self, node: AST.Identifier) -> str:
        """Visit identifier (component reference in clauses)."""
        name = node.value
        if self._clause_context and name in self._clause_context.column_mapping:
            return self._clause_context.column_mapping[name]
        return f'"{name}"'

    # =========================================================================
    # Query Generation
    # =========================================================================

    def _generate_query(self, node: AST.AST) -> str:
        """Generate SQL query for an expression."""
        structure = self._current_query.structure

        # Scalar output
        if isinstance(structure, Scalar):
            return f"SELECT {self.visit(node)} AS value"

        # Check if it's a clause expression (RegularAggregation)
        if isinstance(node, AST.RegularAggregation):
            return self._generate_clause_query(node, structure)

        # Dataset output - use unified JOIN optimization
        return self._generate_dataset_query(node, structure)

    def _generate_clause_query(self, node: AST.RegularAggregation, structure: Dataset) -> str:
        """Generate SQL for clause operations (filter, calc, keep, drop, rename)."""
        # Unwind the clause chain to get base dataset and all clauses in order
        clauses: List[AST.RegularAggregation] = []
        current: AST.AST = node
        while isinstance(current, AST.RegularAggregation):
            clauses.append(current)
            current = current.dataset
        clauses.reverse()  # Now in application order (first clause first)

        # Get base source(s) - could be VarID, BinOp, or other expression
        base_node = current
        base_source, alias, column_mapping = self._resolve_base_source(base_node, structure)

        # Create clause context
        self._clause_context = ClauseContext(
            base_source=base_source, alias=alias, column_mapping=column_mapping
        )

        # Process each clause in order
        for clause in clauses:
            self._process_clause(clause)

        # Build final SQL
        sql = self._build_clause_sql(structure)
        self._clause_context = None
        return sql

    def _resolve_base_source(
        self, node: AST.AST, structure: Dataset
    ) -> Tuple[str, str, Dict[str, str]]:
        """Resolve base source for clause operations. Returns (source, alias, column_mapping)."""
        if isinstance(node, AST.VarID):
            # Simple dataset reference
            name = node.value
            alias = "t0"
            column_mapping = {
                comp: f'{alias}."{comp}"' for comp in self.datasets[name].get_components_names()
            }
            return f'"{name}"', alias, column_mapping

        if isinstance(node, AST.BinOp):
            # Binary operation on datasets - generate subquery
            subquery = self._generate_dataset_query(node, structure)
            alias = "t0"
            column_mapping = {
                comp: f'{alias}."{comp}"' for comp in structure.get_components_names()
            }
            return f"({subquery})", alias, column_mapping

        # Fallback - shouldn't normally happen
        return self.visit(node), "t0", {}

    def _process_clause(self, clause: AST.RegularAggregation) -> None:
        """Process a single clause and update the clause context."""
        ctx = self._clause_context
        if ctx is None:
            return

        op = clause.op.lower() if isinstance(clause.op, str) else str(clause.op)

        if op == "filter":
            # Filter condition
            condition = self.visit(clause.children[0])
            ctx.where_conditions.append(condition)

        elif op == "calc":
            # Calc: create or override columns
            for child in clause.children:
                if isinstance(child, AST.UnaryOp) and child.op in ("measure", "attribute"):
                    # Role setter wrapping assignment
                    assignment = child.operand
                else:
                    assignment = child
                if hasattr(assignment, "left") and hasattr(assignment, "right"):
                    col_name = assignment.left.value
                    expr = self.visit(assignment.right)
                    ctx.calc_expressions[col_name] = expr
                    # Update column mapping for subsequent operations
                    ctx.column_mapping[col_name] = expr

        elif op == "keep":
            # Keep only specified columns (plus identifiers)
            keep_names = {child.value for child in clause.children if hasattr(child, "value")}
            ctx.keep_columns = keep_names

        elif op == "drop":
            # Drop specified columns
            drop_names = {child.value for child in clause.children if hasattr(child, "value")}
            ctx.drop_columns.update(drop_names)

        elif op == "rename":
            # Rename columns
            for child in clause.children:
                if isinstance(child, AST.RenameNode):
                    old_name = child.old_name
                    new_name = child.new_name
                    ctx.rename_mapping[old_name] = new_name
                    # Update column mapping
                    if old_name in ctx.column_mapping:
                        ctx.column_mapping[new_name] = ctx.column_mapping.pop(old_name)

    def _build_clause_sql(self, structure: Dataset) -> str:
        """Build final SQL from clause context."""
        ctx = self._clause_context
        if ctx is None:
            return ""

        # Determine final columns from structure
        select_parts = []
        for comp in structure.get_components():
            name = comp.name
            # Check if this column was renamed
            original_name = None
            for old, new in ctx.rename_mapping.items():
                if new == name:
                    original_name = old
                    break

            # Get expression
            if name in ctx.calc_expressions:
                expr = ctx.calc_expressions[name]
            elif original_name and original_name in ctx.column_mapping:
                expr = ctx.column_mapping[original_name]
            elif name in ctx.column_mapping:
                expr = ctx.column_mapping[name]
            else:
                expr = f'{ctx.alias}."{name}"'

            select_parts.append(f'{expr} AS "{name}"')

        # Build SQL
        sql = f"SELECT {', '.join(select_parts)}\nFROM {ctx.base_source} AS {ctx.alias}"

        if ctx.where_conditions:
            sql += f"\nWHERE {' AND '.join(ctx.where_conditions)}"

        return sql

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

    def _build_from_clause(self, dataset_names: Set[str]) -> Tuple[str, Dict[str, str]]:
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
