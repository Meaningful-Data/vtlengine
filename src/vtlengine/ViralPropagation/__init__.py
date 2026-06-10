"""
ViralPropagation
================

Registry for viral attribute propagation rules as defined by the VTL 2.2
``define viral propagation`` construct. Value resolution is generated as SQL
in :mod:`vtlengine.ViralPropagation.sql`.
"""

from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional

import pandas as pd


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

    def register(self, rule: ViralPropagationRule) -> None:
        """Register a viral propagation rule."""
        if rule.signature_type == "variable":
            self._variable_rules[rule.target] = rule
        else:
            self._valuedomain_rules[rule.target] = rule

    def get_rule_for_variable(self, variable_name: str) -> Optional[ViralPropagationRule]:
        """Get the propagation rule for a variable (variable-level overrides value domain)."""
        if variable_name in self._variable_rules:
            return self._variable_rules[variable_name]
        # Value domain lookup would require knowing which VD the variable uses.
        # For v1, only variable-level rules are supported.
        return None

    def get_existing(self, signature_type: str, target: str) -> Optional["ViralPropagationRule"]:
        """Return an already-registered rule with the same signature_type and target."""
        rules = self._variable_rules if signature_type == "variable" else self._valuedomain_rules
        return rules.get(target)

    def rule_for(self, component: Any) -> Optional["ViralPropagationRule"]:
        """Resolve the rule for a component: variable-level overrides value-domain-level.

        ``component.value_domain`` may not exist yet (added in a later phase); use
        getattr so this is forward-compatible.
        """
        rule = self._variable_rules.get(component.name)
        if rule is not None:
            return rule
        value_domain = getattr(component, "value_domain", None)
        if value_domain is not None:
            return self._valuedomain_rules.get(value_domain)
        return None

    # Pandas resolution methods
    def resolve_pair(self, variable_name: str, value_a: Any, value_b: Any) -> Any:
        """Resolve two viral attribute values into one (for binary operators)."""
        rule = self.get_rule_for_variable(variable_name)
        if rule is None or pd.isna(value_a) or pd.isna(value_b):
            return None

        if rule.aggregate_function is not None:
            if rule.aggregate_function == "avg":
                return (value_a + value_b) / 2
            elif rule.aggregate_function == "min":
                return min(value_a, value_b)
            elif rule.aggregate_function == "max":
                return max(value_a, value_b)
            elif rule.aggregate_function == "sum":
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

    def resolve_group(self, variable_name: str, values: List[Any]) -> Any:
        """Resolve N values (for aggregation/analytic operators)."""
        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]
        rule = self.get_rule_for_variable(variable_name)
        if rule is None:
            return None

        if rule.aggregate_function is not None:
            funcs: Dict[str, Any] = {
                "min": min,
                "max": max,
                "sum": sum,
                "avg": lambda v: sum(v) / len(v),
            }
            return funcs[rule.aggregate_function](values)

        # Enumerated: reduce pairwise (associative + commutative guarantees correctness)
        return reduce(lambda a, b: self.resolve_pair(variable_name, a, b), values)

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
