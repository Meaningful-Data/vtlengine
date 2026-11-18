"""How the values of user defined operators meet the types of their signature (issue #1166)."""

from copy import copy
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, cast

from vtlengine.AST import SetType, SignatureConstraints, TypeConstraint, ValueDomainType
from vtlengine.AST.ASTString import _format_component_type, _format_name, _format_type
from vtlengine.DataTypes import check_unary_implicit_promotion
from vtlengine.Exceptions import InputValidationException, SemanticError
from vtlengine.Model import Component, Dataset, Scalar, ValueDomain


def _value_domain(
    marker: ValueDomainType, value_domains: Optional[Dict[str, ValueDomain]]
) -> ValueDomain:
    """The value domain a signature names, which must be given."""
    if marker.name not in (value_domains or {}):
        # A name that is neither a basic type nor a value domain given
        raise InputValidationException(code="0-5-1-10", definition=marker.definition)
    return cast(Dict[str, ValueDomain], value_domains)[marker.name]


def _on_value_domain(
    constraint: Optional[TypeConstraint], value_domain: ValueDomain
) -> TypeConstraint:
    """The constraint with the values that must also be values of the value domain."""
    return replace(constraint or TypeConstraint(), value_domain=value_domain)


def resolve_value_domain_type(
    declared: Any,
    constraints: Optional[SignatureConstraints],
    value_domains: Optional[Dict[str, ValueDomain]],
) -> Tuple[Any, Optional[SignatureConstraints]]:
    """The type of a parameter or of the result of a user defined operator, with each value
    domain that types it, a set of it or a component of its dataset type replaced by the basic
    type of the value domain, which the values must be values of (issues #1236, #1351). Any
    other type is the one declared."""
    if isinstance(declared, Dataset):
        return _resolve_dataset_type(declared, constraints, value_domains)
    marker = declared.data_type if isinstance(declared, (Component, SetType)) else declared
    if not isinstance(marker, ValueDomainType):
        return declared, constraints
    value_domain = _value_domain(marker, value_domains)
    if isinstance(declared, Component):
        declared = replace(declared, data_type=value_domain.type, value_domain=value_domain)
    elif isinstance(declared, SetType):
        declared = SetType(value_domain.type)
    else:
        declared = value_domain.type
    constraints = constraints or SignatureConstraints()
    return declared, replace(constraints, value=_on_value_domain(constraints.value, value_domain))


def _resolve_dataset_type(
    declared: Dataset,
    constraints: Optional[SignatureConstraints],
    value_domains: Optional[Dict[str, ValueDomain]],
) -> Tuple[Any, Optional[SignatureConstraints]]:
    """A dataset type with the components that a value domain types on its basic type."""
    original = constraints
    constraints = constraints or SignatureConstraints()
    components, named = dict(declared.components), dict(constraints.components)
    for name, component in declared.components.items():
        if isinstance(component.data_type, ValueDomainType):
            value_domain = _value_domain(component.data_type, value_domains)
            components[name] = replace(component, data_type=value_domain.type)
            named[name] = _on_value_domain(named.get(name), value_domain)
    unnamed = []
    for entry in constraints.unnamed:
        if isinstance(entry.component.data_type, ValueDomainType):
            value_domain = _value_domain(entry.component.data_type, value_domains)
            component = replace(entry.component, data_type=value_domain.type)
            constraint = _on_value_domain(entry.constraint, value_domain)
            entry = replace(entry, component=component, constraint=constraint)
        unnamed.append(entry)
    if components == declared.components and unnamed == constraints.unnamed:
        return declared, original
    resolved = SignatureConstraints(constraints.value, named, unnamed)
    return replace(declared, components=components), resolved


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
                declared=_format_component_type(comp, named.get(name)),
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
                component=_format_component_type(entry.component, entry.constraint),
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


def udo_result_on_value_domains(
    result: Any,
    constraints: Optional[SignatureConstraints],
    taken: Dict[str, Optional[TypeConstraint]],
) -> Any:
    """The result of a user defined operator, whose values are values of the value domains its
    returns names, defined on them: the scalar or Component result, or the components that the
    dataset type names on one, which take the basic type of the value domain (issue #1351)."""
    if isinstance(result, Dataset):
        domains = {
            name: constraint.value_domain
            for name, constraint in taken.items()
            if constraint is not None and constraint.value_domain is not None
        }
        if not domains:
            return result
        components = dict(result.components)
        for name, domain in domains.items():
            components[name] = copy(components[name])
            components[name].data_type, components[name].value_domain = domain.type, domain
        return Dataset(name=result.name, components=components, data=result.data)
    constraint = constraints.value if constraints is not None else None
    if constraint is None or constraint.value_domain is None:
        return result
    domain = constraint.value_domain
    result = copy(result)
    result.data_type, result.value_domain = domain.type, domain
    if isinstance(result, Scalar):
        result.value = domain.type.cast(result.value)
    return result
