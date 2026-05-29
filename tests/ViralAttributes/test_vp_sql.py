"""The generated SQL must match the resolve_pair / resolve_group reference oracle."""

import duckdb
import pytest

from vtlengine.ViralPropagation import ViralPropagationRegistry, ViralPropagationRule
from vtlengine.ViralPropagation.sql import vp_group_sql, vp_pair_sql

CONF = ViralPropagationRule(
    name="CONF",
    signature_type="variable",
    target="v",
    enumerated_clauses=[{"values": ["C"], "result": "C"}, {"values": ["N"], "result": "N"}],
    default_value="F",
)
COMP = ViralPropagationRule(
    name="COMP",
    signature_type="variable",
    target="v",
    enumerated_clauses=[{"values": ["C", "M"], "result": "N"}, {"values": ["M"], "result": "M"}],
    default_value=" ",
)
SMAX = ViralPropagationRule(
    name="S", signature_type="variable", target="v", aggregate_function="max"
)


def _reg(rule: ViralPropagationRule) -> ViralPropagationRegistry:
    reg = ViralPropagationRegistry()
    reg.register(rule)
    return reg


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@pytest.mark.parametrize(
    "rule,a,b",
    [
        (CONF, "C", "N"),
        (CONF, "N", "F"),
        (CONF, "F", "F"),
        (CONF, "X", "Y"),
        (COMP, "C", "M"),
        (COMP, "M", "C"),
        (COMP, "M", "F"),
        (COMP, "X", "Y"),
        (SMAX, "C", "N"),
    ],
)
def test_pair_sql_matches_oracle(rule: ViralPropagationRule, a: str, b: str) -> None:
    con = duckdb.connect()
    sql = vp_pair_sql(rule, _lit(a), _lit(b))
    got = con.execute(f"SELECT {sql}").fetchone()[0]
    assert got == _reg(rule).resolve_pair("v", a, b)


@pytest.mark.parametrize(
    "rule,vals",
    [
        (CONF, ["C", "N", "F"]),
        (CONF, ["N", "F"]),
        (CONF, ["F"]),
        (COMP, ["C", "M", "X"]),
        (SMAX, ["C", "N", "F"]),
    ],
)
def test_group_sql_matches_oracle(rule: ViralPropagationRule, vals: list) -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE g(v VARCHAR)")
    con.executemany("INSERT INTO g VALUES (?)", [(x,) for x in vals])
    sql = vp_group_sql(rule, "v")
    got = con.execute(f"SELECT {sql} FROM g").fetchone()[0]
    assert got == _reg(rule).resolve_group("v", vals)


class _FakeComp:
    def __init__(self, name: str, value_domain: object = None) -> None:
        self.name = name
        self.value_domain = value_domain


def test_rule_for_variable_then_value_domain() -> None:
    reg = ViralPropagationRegistry()
    var_rule = ViralPropagationRule(
        name="v", signature_type="variable", target="VAt_1", aggregate_function="max"
    )
    vd_rule = ViralPropagationRule(
        name="d", signature_type="valuedomain", target="CL_X", aggregate_function="min"
    )
    reg.register(var_rule)
    reg.register(vd_rule)
    # variable-level wins over value-domain-level for the same attribute
    assert reg.rule_for(_FakeComp("VAt_1", value_domain="CL_X")) is var_rule
    # value-domain fallback when no variable rule
    assert reg.rule_for(_FakeComp("VAt_2", value_domain="CL_X")) is vd_rule
    # nothing matches
    assert reg.rule_for(_FakeComp("VAt_3", value_domain="CL_Y")) is None
    # component without the attribute present (forward-compatible)
    assert reg.rule_for(_FakeComp("VAt_3")) is None
