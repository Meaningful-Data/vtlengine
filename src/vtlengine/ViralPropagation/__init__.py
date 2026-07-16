"""
ViralPropagation
================

Registry for viral attribute propagation rules as defined by the VTL 2.2
``define viral propagation`` construct. Value resolution is generated as SQL
in :mod:`vtlengine.ViralPropagation.sql`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


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


def require_rules(components: Iterable[Any]) -> None:
    """Raise SemanticError 1-3-3-6 for any viral component lacking a propagation rule.

    Call this at the combination points defined by the VTL 2.2 attribute propagation
    rule (an operation over two or more datasets, an aggregation/analytic group-by, or
    a hierarchy roll-up), where the default propagation algorithm must be executed. A
    viral attribute that is only copied through (row-preserving operators) needs no rule.
    """
    from vtlengine.Exceptions import SemanticError  # local import avoids an import cycle

    registry = get_current_registry()
    for comp in components:
        if registry.rule_for(comp) is None:
            raise SemanticError("1-3-3-6", name=comp.name)


def combined_viral_components(operands: Iterable[Any]) -> List[Any]:
    """Return the viral components combined across ``operands``.

    A viral attribute whose data points are combined appears (by name) as a viral
    attribute in two or more operands; those require a propagation rule. A viral
    attribute present in a single operand is copied through and needs no rule.
    """
    counts: Dict[str, int] = {}
    comp_by_name: Dict[str, Any] = {}
    for op in operands:
        for comp in op.get_viral_attributes():
            counts[comp.name] = counts.get(comp.name, 0) + 1
            comp_by_name[comp.name] = comp
    return [comp_by_name[name] for name, n in counts.items() if n >= 2]
