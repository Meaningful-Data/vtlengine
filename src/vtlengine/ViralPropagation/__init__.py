"""
ViralPropagation
================

Registry and resolution logic for viral attribute propagation rules
as defined by VTL 2.2 ``define viral propagation`` construct.
"""

from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional


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

    def resolve_pair(self, variable_name: str, value_a: Any, value_b: Any) -> Any:
        """Resolve two viral attribute values into one (for binary operators)."""
        rule = self.get_rule_for_variable(variable_name)
        if rule is None:
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

        pair = {value_a, value_b}
        for clause in binary_clauses:
            if set(clause["values"]) == pair:
                return clause["result"]

        for clause in unary_clauses:
            if clause["values"][0] in pair:
                return clause["result"]

        return rule.default_value

    def resolve_group(self, variable_name: str, values: List[Any]) -> Any:
        """Resolve N values (for aggregation/analytic operators)."""
        rule = self.get_rule_for_variable(variable_name)
        if rule is None:
            return None

        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]

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
