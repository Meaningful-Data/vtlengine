"""
ViralPropagation
================

Registry for viral attribute propagation rules as defined by the VTL 2.2
``define viral propagation`` construct. Value resolution is generated as SQL
in :mod:`vtlengine.ViralPropagation.sql`.
"""

from dataclasses import dataclass, field
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
