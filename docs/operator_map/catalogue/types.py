"""The data types of the VTL 2.2 type system, the hubs of the type flows of the map.

A basic type has one variant, ``identity``: a dataset with a measure of the type, assigned to
another dataset. It shows that vtlengine loads, keeps and writes the values of the type. The
abstract ``scalar`` and the structural hubs (``dataset``, ``set``, ``value_domain``, the
rulesets, ``viral_propagation``, ``operator`` and ``external_routine``) have no example of their
own: they link to the operator that introduces them (``TypeDef.see``).
"""

from __future__ import annotations

from tool.model import TypeDef, Variant, authored


def _identity() -> tuple[Variant, ...]:
    return (Variant("identity", "DS_r := DS_1", (authored(),)),)


TYPES: tuple[TypeDef, ...] = (
    TypeDef("scalar", (), see="general.non_persistent_assignment"),
    TypeDef("number", ("scalar",), _identity()),
    TypeDef("integer", ("number",), _identity()),
    TypeDef("string", ("scalar",), _identity()),
    TypeDef("boolean", ("scalar",), _identity()),
    TypeDef("time", ("scalar",), _identity()),
    TypeDef("date", ("time",), _identity()),
    TypeDef("time_period", ("time",), _identity()),
    TypeDef("duration", ("scalar",), _identity()),
    TypeDef("dataset", (), see="general.non_persistent_assignment"),
    TypeDef("set", (), see="comparison.in"),
    TypeDef("value_domain", (), see="comparison.in"),
    TypeDef("datapoint_ruleset", (), see="definitions.define_datapoint_ruleset"),
    TypeDef("hierarchical_ruleset", (), see="definitions.define_hierarchical_ruleset"),
    TypeDef("viral_propagation", (), see="definitions.define_viral_propagation"),
    TypeDef("operator", (), see="definitions.define_operator"),
    TypeDef("external_routine", (), see="general.eval"),
)
