"""DuckDB SQL generation for viral attribute propagation rules.

These functions turn a registered ``ViralPropagationRule`` into SQL fragments
that combine viral-attribute values per the rule (enumerated when/then clauses
or an aggregate function). This module imports nothing from the transpiler or
the data model, so it is free of import cycles.
"""

from typing import Any, Callable, Dict, List, Optional

from vtlengine.ViralPropagation import ViralPropagationRule

# Binary (two-operand) forms of the aggregate functions.
_AGG_BINARY: Dict[str, Callable[[str, str], str]] = {
    "min": lambda a, b: f"LEAST({a}, {b})",
    "max": lambda a, b: f"GREATEST({a}, {b})",
    "sum": lambda a, b: f"({a} + {b})",
    "avg": lambda a, b: f"(({a} + {b}) / 2.0)",
}
# Group (N-operand) native aggregates.
_AGG_GROUP: Dict[str, str] = {"min": "MIN", "max": "MAX", "sum": "SUM", "avg": "AVG"}


def _sql_literal(value: Optional[Any]) -> str:
    """Render an enumerated clause value as a single-quoted SQL string literal."""
    if value is None:
        return "NULL"
    return "'" + value.replace("'", "''") + "'"


def _value_in_pair(value: Optional[Any], a_ref: str, b_ref: str) -> str:
    """SQL predicate: the (possibly null) clause value is one of the two operand refs."""
    if value is None:
        return f"({a_ref} IS NULL OR {b_ref} IS NULL)"
    return f"{_sql_literal(value)} IN ({a_ref}, {b_ref})"


def _enumerated_case(rule: ViralPropagationRule, a_ref: str, b_ref: str) -> str:
    """A CASE expr: binary clauses first, then unary, then default.

    Assumes the two values of a binary clause are distinct (guaranteed by the grammar +
    the upstream duplicate-combination check), so ``v1 IN (a,b) AND v2 IN (a,b)`` matches
    set-equality semantics.
    When there are no WHEN clauses (e.g. an else-only rule), emits just the default
    scalar expression instead of the invalid ``CASE  ELSE ... END`` form.
    """
    whens: List[str] = []
    binary = [c for c in rule.enumerated_clauses if len(c["values"]) == 2]
    unary = [c for c in rule.enumerated_clauses if len(c["values"]) == 1]
    for clause in binary:
        cond_a = _value_in_pair(clause["values"][0], a_ref, b_ref)
        cond_b = _value_in_pair(clause["values"][1], a_ref, b_ref)
        whens.append(f"WHEN {cond_a} AND {cond_b} THEN {_sql_literal(clause['result'])}")
    for clause in unary:
        cond = _value_in_pair(clause["values"][0], a_ref, b_ref)
        whens.append(f"WHEN {cond} THEN {_sql_literal(clause['result'])}")
    default = _sql_literal(rule.default_value)
    if not whens:
        return default
    return "CASE " + " ".join(whens) + f" ELSE {default} END"


def vp_pair_sql(rule: ViralPropagationRule, a_ref: str, b_ref: str) -> str:
    """SQL expression combining two viral values for a binary operator."""
    if rule.aggregate_function is not None:
        return _AGG_BINARY[rule.aggregate_function](a_ref, b_ref)
    return _enumerated_case(rule, a_ref, b_ref)


def vp_reduce_refs(rule: ViralPropagationRule, refs: List[str]) -> str:
    """Fold vp_pair_sql across an ordered list of column refs (>=1)."""
    if not refs:
        raise ValueError("vp_reduce_refs requires at least one column ref")
    acc = refs[0]
    for ref in refs[1:]:
        acc = vp_pair_sql(rule, acc, ref)
    return acc


def vp_group_sql(rule: ViralPropagationRule, col_ref: str) -> str:
    """SQL aggregate expression combining a group of viral values."""
    if rule.aggregate_function is not None:
        return f"{_AGG_GROUP[rule.aggregate_function]}({col_ref})"
    case = _enumerated_case(rule, "acc", "x")
    return f"list_reduce(list({col_ref}), (acc, x) -> {case})"


def vp_group_sql_windowed(rule: ViralPropagationRule, col_ref: str, over_clause: str) -> str:
    """Windowed form of vp_group_sql for analytic invocation (... OVER (window))."""
    if rule.aggregate_function is not None:
        return f"{_AGG_GROUP[rule.aggregate_function]}({col_ref}) OVER ({over_clause})"
    case = _enumerated_case(rule, "acc", "x")
    return f"list_reduce(list({col_ref}) OVER ({over_clause}), (acc, x) -> {case})"


def vp_no_rule_group_sql(col_ref: str) -> str:
    """Group no-rule keep: copy the value of a single-row group, else NULL."""
    return f"CASE WHEN COUNT(*) = 1 THEN MAX({col_ref}) ELSE NULL END"
