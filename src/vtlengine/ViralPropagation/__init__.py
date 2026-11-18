"""
ViralPropagation
================

Registry for viral attribute propagation rules as defined by the VTL 2.2
``define viral propagation`` construct. Value resolution is generated as SQL
in :mod:`vtlengine.ViralPropagation.sql`.
"""

from copy import copy
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import pandas as pd

from vtlengine.DataTypes import (
    BASIC_TYPES,
    Duration,
    Integer,
    Null,
    Number,
    ScalarType,
    String,
    TimeInterval,
    binary_implicit_promotion,
    check_binary_implicit_promotion,
    check_unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import value_of


@dataclass
class ViralPropagationRule:
    """A single viral propagation rule definition."""

    name: str
    signature_type: str  # "valuedomain" or "variable"
    target: str
    enumerated_clauses: List[Dict[str, Any]] = field(default_factory=list)
    aggregate_function: Optional[str] = None  # "min", "max", "sum", "avg"
    default_value: Optional[str] = None


class ViralPropagationRegistry:
    """Registry that stores viral propagation rules and resolves attribute values."""

    def __init__(self) -> None:
        self._variable_rules: Dict[str, ViralPropagationRule] = {}
        self._valuedomain_rules: Dict[str, ViralPropagationRule] = {}
        # The rule with its constants read as values of a type, by rule and type, so the
        # pandas engine, which resolves a rule per row, reads them once (rule_for)
        self._typed_rules: Dict[
            Tuple[int, Any], Tuple[ViralPropagationRule, ViralPropagationRule]
        ] = {}

    def register(self, rule: ViralPropagationRule) -> None:
        """Register a viral propagation rule."""
        if rule.signature_type == "variable":
            self._variable_rules[rule.target] = rule
        else:
            self._valuedomain_rules[rule.target] = rule

    def get_existing(self, signature_type: str, target: str) -> Optional["ViralPropagationRule"]:
        """Return an already-registered rule with the same signature_type and target."""
        rules = self._variable_rules if signature_type == "variable" else self._valuedomain_rules
        return rules.get(target)

    def rule_for(self, component: Any) -> Optional["ViralPropagationRule"]:
        """Resolve the rule for a component: a rule on the variable overrides one on the value
        domain the component is defined on. The constants of the rule are read as values of
        the type of the component, whatever form they are written in: "2024Q1" is the
        Time_Period 2024-Q1 (issue #1349)."""
        rule = self._variable_rules.get(component.name)
        value_domain = getattr(component, "value_domain", None)
        if rule is None and value_domain is not None:
            rule = self._valuedomain_rules.get(value_domain.name)
        if rule is None or not rule.enumerated_clauses:
            return rule
        data_type = component.data_type
        cached = self._typed_rules.get((id(rule), data_type))
        if cached is not None and cached[0] is rule:
            return cached[1]
        clauses = [
            {
                "values": [value_of(data_type, value) for value in clause["values"]],
                "result": value_of(data_type, clause["result"]),
            }
            for clause in rule.enumerated_clauses
        ]
        typed_rule = replace(
            rule,
            enumerated_clauses=clauses,
            default_value=value_of(data_type, rule.default_value),
        )
        # The rule itself is kept next to it, so its id is not reused by another rule
        self._typed_rules[(id(rule), data_type)] = (rule, typed_rule)
        return typed_rule

    # Pandas resolution methods, on the viral attribute of the operand whose values combine
    def resolve_pair(self, component: Any, value_a: Any, value_b: Any) -> Any:
        """Resolve two viral attribute values into one (for binary operators)."""
        rule = self.rule_for(component)
        if rule is None:
            return None

        if rule.aggregate_function is not None:
            fn = rule.aggregate_function
            if fn in ("min", "max"):
                if pd.isna(value_a):
                    return None if pd.isna(value_b) else value_b
                if pd.isna(value_b):
                    return value_a
                return min(value_a, value_b) if fn == "min" else max(value_a, value_b)
            elif fn == "avg":
                return (value_a + value_b) / 2
            elif fn == "sum":
                return value_a + value_b
            return None

        # Enumerated: binary clauses first, then unary (per spec)
        binary_clauses = [c for c in rule.enumerated_clauses if len(c["values"]) == 2]
        unary_clauses = [c for c in rule.enumerated_clauses if len(c["values"]) == 1]

        pair = {self._normalize_null(value_a), self._normalize_null(value_b)}
        for clause in binary_clauses:
            if {self._normalize_null(v) for v in clause["values"]} == pair:
                return clause["result"]

        for clause in unary_clauses:
            if self._normalize_null(clause["values"][0]) in pair:
                return clause["result"]

        return rule.default_value

    def resolve_single(self, component: Any, value: Any) -> Any:
        """Resolve a single viral value through the rule (row-preserving operators).

        Enumerated rules apply the matching single-value clause, else the default;
        aggregate rules and "no rule" leave a lone value unchanged.
        """
        rule = self.rule_for(component)
        if rule is None or rule.aggregate_function is not None:
            return value
        for clause in rule.enumerated_clauses:
            if len(clause["values"]) == 1 and self._normalize_null(
                clause["values"][0]
            ) == self._normalize_null(value):
                return clause["result"]
        return rule.default_value

    def resolve_group(self, component: Any, values: List[Any]) -> Any:
        """Resolve N values (for aggregation/analytic operators)."""
        if len(values) == 0:
            return None
        if len(values) == 1:
            return self.resolve_single(component, values[0])
        rule = self.rule_for(component)
        if rule is None:
            return None

        if rule.aggregate_function is not None:
            # Native SQL aggregates (MIN/MAX/SUM/AVG) ignore nulls; mirror that here so
            # the pandas path matches DuckDB and min/max do not choke on pd.NA.
            non_null = [v for v in values if not pd.isna(v)]
            if not non_null:
                return None
            funcs: Dict[str, Any] = {
                "min": min,
                "max": max,
                "sum": sum,
                "avg": lambda v: sum(v) / len(v),
            }
            return funcs[rule.aggregate_function](non_null)

        # Enumerated: reduce pairwise (associative + commutative guarantees correctness)
        return reduce(lambda a, b: self.resolve_pair(component, a, b), values)

    def clear(self) -> None:
        """Clear all registered rules."""
        self._variable_rules.clear()
        self._valuedomain_rules.clear()

    def _normalize_null(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        return value


# Module-level accessor for operators to use.
# The Interpreter sets this at the start of each run() call.
_current_registry: Optional[ViralPropagationRegistry] = None


def get_current_registry() -> ViralPropagationRegistry:
    """Get the current viral propagation registry."""
    global _current_registry  # noqa: PLW0603
    if _current_registry is None:
        _current_registry = ViralPropagationRegistry()
    return _current_registry


def set_current_registry(registry: ViralPropagationRegistry) -> None:
    """Set the current viral propagation registry (called by Interpreter)."""
    global _current_registry  # noqa: PLW0603
    _current_registry = registry


def _is_value_of(value: Any, data_type: Type[ScalarType]) -> bool:
    """Check that a rule constant is a value of ``data_type``. A temporal value can only be
    written as a string, any other constant must have a type promotable to ``data_type``."""
    value_type = BASIC_TYPES[type(value)]
    if issubclass(data_type, (TimeInterval, Duration)):
        compatible = value_type in (String, Null)
    else:
        compatible = check_unary_implicit_promotion(value_type, data_type)
    return compatible and data_type.check(value)


def is_value_of_domain(value: Any, value_domain: Any) -> bool:
    """Check that a rule constant is a value of the value domain the rule is defined on: a
    null, or a value of its basic type that is one of its values (issue #1349)."""
    values = pd.Series([value], dtype=object)
    return _is_value_of(value, value_domain.type) and not len(value_domain.outside(values))


def check_duplicate_clauses(name: str, clauses: Iterable[List[Any]]) -> None:
    """Raise SemanticError 1-3-3-4 for two clauses of a rule on the same values, compared as
    values of their type when the caller has read them so (issue #1349)."""
    seen: List[Any] = []
    for values in clauses:
        key = frozenset(values)
        if key in seen:
            raise SemanticError("1-3-3-4", values=values, name=name)
        seen.append(key)


def require_rules(components: Iterable[Any]) -> None:
    """Raise SemanticError 1-3-3-6 for any viral component lacking a propagation rule.

    Call this at the combination points defined by the VTL 2.2 attribute propagation
    rule (an operation over two or more datasets, an aggregation/analytic group-by, or
    a hierarchy roll-up), where the default propagation algorithm must be executed. A
    viral attribute that is only copied through (row-preserving operators) needs no rule.

    Also raises SemanticError 1-3-3-5 when a ``sum``/``avg`` rule applies to a
    non-numeric viral attribute: the check runs only where the rule actually executes,
    so a non-numeric attribute that is never combined raises nothing (issue #910).
    Likewise raises SemanticError 1-3-3-9 when a constant of an enumerated rule is not a
    value of the viral attribute's data type.
    """
    registry = get_current_registry()
    for comp in components:
        rule = registry.rule_for(comp)
        if rule is None:
            raise SemanticError("1-3-3-6", name=comp.name)
        if rule.aggregate_function in ("sum", "avg") and comp.data_type not in (Integer, Number):
            raise SemanticError(
                "1-3-3-5",
                name=comp.name,
                function=rule.aggregate_function,
                type=comp.data_type.__name__,
            )
        check_duplicate_clauses(rule.name, (c["values"] for c in rule.enumerated_clauses))
        constants = [v for c in rule.enumerated_clauses for v in (*c["values"], c["result"])]
        for value in [*constants, rule.default_value]:
            if not _is_value_of(value, comp.data_type):
                raise SemanticError(
                    "1-3-3-9",
                    rule=rule.name,
                    name=comp.name,
                    value=repr(value),
                    type=comp.data_type.__name__,
                )


def _common_type(type_a: Any, type_b: Any) -> Any:
    """The common type of two types of one viral attribute, or None when they have none."""
    if type_a == type_b:
        return type_a
    if check_binary_implicit_promotion(type_a, type_b):
        return binary_implicit_promotion(type_a, type_b)
    return None


def apply_viral_return_types(components: Iterable[Any], result_components: Dict[str, Any]) -> None:
    """Set the result data type of viral attributes combined under an aggregate rule:
    ``avg`` gives Number, as it yields fractional values, mirroring the Avg operator's
    return type, and min/max/sum give the type of the combined component, the common
    type of the operands that :func:`combined_viral_components` gives (sum mirrors Sum,
    which has no return type) (issue #1314). Enumerated rules keep the input type. A rule on
    a value domain keeps it with min and max, which pick one of the values, while sum and avg
    compute new values, of the basic type (issue #1348). Promoted entries are always replaced
    with a copy, because some operators share the result component objects (or the dict
    itself) with their input datasets and the input structure must never be mutated.

    Call this right after :func:`require_rules` at every combination point, with the
    same combined components and the result-components mapping under construction.
    """
    registry = get_current_registry()
    for comp in components:
        result_comp = result_components.get(comp.name)
        if result_comp is None:
            continue
        rule = registry.rule_for(comp)
        aggregate = rule.aggregate_function if rule is not None else None
        on_domain = rule is not None and rule.signature_type == "valuedomain"
        if on_domain and aggregate not in ("sum", "avg"):
            continue
        if aggregate is None and result_comp.value_domain is None:
            continue
        promoted = copy(result_comp)
        promoted.value_domain = None
        if aggregate == "avg":
            promoted.data_type = Number
        elif aggregate is not None:
            common = _common_type(result_comp.data_type, comp.data_type)
            promoted.data_type = result_comp.data_type if common is None else common
        result_components[comp.name] = promoted


def combine_viral_groups(
    data: pd.DataFrame, keys: List[str], components: List[Any]
) -> pd.DataFrame:
    """The viral attributes ``components`` of each group of data points by ``keys``, combined by
    their propagation rules: one row per group, or a single row when there are no keys."""
    registry = get_current_registry()
    if not keys:
        return pd.DataFrame(
            {
                comp.name: [registry.resolve_group(comp, list(data[comp.name]))]
                for comp in components
            }
        )
    grouped = data.groupby(keys, sort=False)

    def _group_resolver(component: Any) -> Any:
        # Binds the attribute per column: a lambda in the comprehension below would see the
        # last one only.
        return lambda values: registry.resolve_group(component, list(values))

    return pd.DataFrame(
        {comp.name: grouped[comp.name].agg(_group_resolver(comp)) for comp in components}
    ).reset_index()


def combined_viral_components(operands: Iterable[Any]) -> List[Any]:
    """Return the viral components combined across ``operands``.

    A viral attribute whose data points are combined appears (by name) as a viral
    attribute in two or more operands; those require a propagation rule. A viral
    attribute present in a single operand is copied through and needs no rule. The
    component returned has the common type of the operands, when their types have one
    (issue #1314), and their value domain only when all of them are on it (issue #1350).
    """
    counts: Dict[str, int] = {}
    comp_by_name: Dict[str, Any] = {}
    for op in operands:
        for comp in op.get_viral_attributes():
            counts[comp.name] = counts.get(comp.name, 0) + 1
            seen = comp_by_name.get(comp.name)
            comp_by_name[comp.name] = comp if seen is None else combine_viral_component(seen, comp)
    return [comp_by_name[name] for name, n in counts.items() if n >= 2]


def operands_rule(name: str, operands: Iterable[Any]) -> Optional[ViralPropagationRule]:
    """The rule that combines the viral attribute ``name`` of the Data Set ``operands``: the rule
    of its Components in them, combined, and not of the result Component, which an ``avg`` rule
    has left without their value domain (issue #1350)."""
    comps = [op.components[name] for op in operands if name in op.components]
    if not comps:
        return None
    return get_current_registry().rule_for(reduce(combine_viral_component, comps))


def _domain_name(component: Any) -> Optional[str]:
    """The name of the value domain ``component`` is defined on, or None."""
    return None if component.value_domain is None else component.value_domain.name


def combine_viral_component(comp_a: Any, comp_b: Any) -> Any:
    """The viral attribute ``comp_b`` as it is combined with ``comp_a``, of another operand: of
    their common type, when they have one, and on their value domain only when both are on it.
    So a rule on a value domain applies only when every operand is on it, and otherwise only a
    rule on the variable does, in any order of the operands (issue #1350)."""
    common = _common_type(comp_a.data_type, comp_b.data_type)
    same_domain = _domain_name(comp_a) == _domain_name(comp_b)
    if (common is None or common == comp_b.data_type) and same_domain:
        return comp_b
    combined = copy(comp_b)
    if common is not None:
        combined.data_type = common
    if not same_domain:
        combined.value_domain = None
    return combined
