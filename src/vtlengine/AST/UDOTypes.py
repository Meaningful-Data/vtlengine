"""How the values of user defined operators meet the types of their signature (issue #1166)."""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, cast

from vtlengine.AST import SignatureConstraints, TypeConstraint, ValueDomainType
from vtlengine.AST.ASTString import _format_component_type, _format_name, _format_type
from vtlengine.DataTypes import check_unary_implicit_promotion
from vtlengine.Exceptions import InputValidationException, SemanticError
from vtlengine.Model import Component, Dataset, ValueDomain


def resolve_value_domain_type(
    declared: Any,
    constraints: Optional[SignatureConstraints],
    value_domains: Optional[Dict[str, ValueDomain]],
) -> Tuple[Any, Optional[SignatureConstraints]]:
    """The type of a parameter that a value domain types, scalar or component: the basic type of
    the value domain, which the values must be values of. Any other type is the
    one declared."""
    marker = declared.data_type if isinstance(declared, Component) else declared
    if not isinstance(marker, ValueDomainType):
        return declared, constraints
    if marker.name not in (value_domains or {}):
        # A name that is neither a basic type nor a value domain given
        raise InputValidationException(code="0-5-1-10", definition=marker.definition)
    value_domain = cast(Dict[str, ValueDomain], value_domains)[marker.name]
    if isinstance(declared, Component):
        declared = replace(declared, data_type=value_domain.type, value_domain=value_domain)
    else:
        declared = value_domain.type
    constraints = constraints or SignatureConstraints()
    value = replace(constraints.value or TypeConstraint(), value_domain=value_domain)
    return declared, replace(constraints, value=value)


def udo_component_matches(declared: Any, found: Any) -> bool:
    """Whether a component has the role and the type a user defined operator declares."""
    # A component declared without a role or a type is parsed with None and a String instance,
    # so declared is not typed as a Component
    return (declared.role is None or found.role == declared.role) and (
        not isinstance(declared.data_type, type)
        or check_unary_implicit_promotion(found.data_type, declared.data_type)
    )


def udo_dataset_components(
    op: str,
    element: str,
    declared: Dataset,
    constraints: Optional[SignatureConstraints],
    found: Dataset,
) -> Dict[str, Optional[TypeConstraint]]:
    """The components of found, the Data Set of element (a parameter or the output of op), that
    a dataset type takes, each with the constraint on its values. Raises when found does not have
    the structure the type gives.

    The type restricts only the roles it names: their components are the ones it names and
    the ones its _ (one), _+ (at least one) and _* (any number) components take.
    """
    unnamed = constraints.unnamed if constraints is not None else []
    named = constraints.components if constraints is not None else {}
    type_ = _format_type(declared, constraints)
    entries = [*declared.components.values(), *(entry.component for entry in unnamed)]
    roles = {entry.role for entry in entries}
    remaining = {
        name: comp for name, comp in found.components.items() if None in roles or comp.role in roles
    }
    taken: Dict[str, Optional[TypeConstraint]] = {}
    for name, comp in declared.components.items():
        if name not in remaining:
            raise SemanticError("1-3-1-11", op=op, element=element, name=name, type=type_)
        if not udo_component_matches(comp, remaining[name]):
            raise SemanticError(
                "1-3-1-12",
                op=op,
                element=element,
                name=name,
                found=_format_component_type(remaining[name]),
                declared=_format_component_type(comp),
                type=type_,
            )
        del remaining[name]
        taken[name] = named.get(name)
    for entry in unnamed:
        matching = [n for n, c in remaining.items() if udo_component_matches(entry.component, c)]
        if entry.multiplicity == "":
            matching = matching[:1]
        if not matching and entry.multiplicity != "*":
            raise SemanticError(
                "1-3-1-13",
                op=op,
                element=element,
                component=_format_component_type(entry.component),
                modifier=f"_{entry.multiplicity}",
                type=type_,
            )
        for name in matching:
            del remaining[name]
            taken[name] = entry.constraint
    if remaining:
        components = ", ".join(
            f"{_format_component_type(comp)} {_format_name(name)}"
            for name, comp in remaining.items()
        )
        raise SemanticError("1-3-1-14", op=op, element=element, components=components, type=type_)
    return taken
