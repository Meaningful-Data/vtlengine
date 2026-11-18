"""Scalars defined on a value domain (GH #1328, #1349).

https://github.com/Meaningful-Data/vtlengine/issues/1328

A scalar type can name a value domain, as a Component type can (GH #1301). The scalar takes
the basic type of the value domain as its data type, its value must be one of the values of
the value domain, and the operators keep the value domain as they keep the one of a
Component.
"""

import pandas as pd
import pytest

from tests.Helper import engine_params
from vtlengine import run, semantic_analysis
from vtlengine.DataTypes import String
from vtlengine.Exceptions import InputValidationException, RunTimeError
from vtlengine.Model import Scalar, ValueDomain

VALUE_DOMAINS = [
    ValueDomain(name="VD_1", type=String, setlist=["A", "B"]),
    ValueDomain(name="VD_2", type=String, setlist=["A", "X"]),
]

DATA_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "VD_1", "role": "Measure", "nullable": True},
            ],
        }
    ],
    "scalars": [
        {"name": "SC_1", "type": "VD_1"},
        {"name": "SC_2", "type": "String"},
        {"name": "SC_3", "type": "VD_2"},
    ],
}


def _run(script, scalar_values, use_duckdb):
    datapoints = {"DS_1": pd.DataFrame({"Id_1": ["1", "2"], "Me_1": ["A", None]})}
    return run(
        script=script,
        data_structures=DATA_STRUCTURES,
        datapoints=datapoints,
        value_domains=VALUE_DOMAINS,
        scalar_values=scalar_values,
        use_duckdb=use_duckdb,
    )


scalar_result_params = [
    pytest.param("SC_r <- SC_1;", {"SC_1": "A"}, "VD_1", "A", id="assignment"),
    pytest.param("SC_r <- SC_1;", {"SC_1": None}, "VD_1", None, id="null"),
    pytest.param(
        'SC_r <- if SC_2 = "x" then SC_1 else SC_1;',
        {"SC_1": "A", "SC_2": "x"},
        "VD_1",
        "A",
        id="if_same_value_domain",
    ),
    pytest.param(
        'SC_r <- if SC_2 = "x" then SC_1 else "B";',
        {"SC_1": "A", "SC_2": "y"},
        "String",
        "B",
        id="if_basic_type",
    ),
    pytest.param(
        "SC_r <- nvl(SC_1, SC_2);", {"SC_1": None, "SC_2": "B"}, "String", "B", id="nvl_basic_type"
    ),
    pytest.param(
        'SC_r <- case when SC_2 = "x" then SC_1 else "A";',
        {"SC_1": "B", "SC_2": "x"},
        "String",
        "B",
        id="case_basic_type",
    ),
    pytest.param(
        'SC_r <- if SC_2 = "x" then SC_1 else SC_2;',
        {"SC_1": "A", "SC_2": "y"},
        "String",
        "y",
        id="if_outside_value_domain",
    ),
    pytest.param(
        "SC_r <- nvl(SC_1, SC_2);",
        {"SC_1": None, "SC_2": "Q"},
        "String",
        "Q",
        id="nvl_outside_value_domain",
    ),
    pytest.param(
        'SC_r <- case when SC_2 = "x" then SC_1 else SC_2;',
        {"SC_1": "A", "SC_2": "Q"},
        "String",
        "Q",
        id="case_outside_value_domain",
    ),
    pytest.param(
        'SC_r <- if SC_2 = "x" then SC_1 else SC_3;',
        {"SC_1": "A", "SC_2": "y", "SC_3": "X"},
        "String",
        "X",
        id="two_value_domains",
    ),
    pytest.param("SC_r <- cast(SC_2, VD_1);", {"SC_2": "B"}, "VD_1", "B", id="cast"),
    pytest.param('SC_r <- SC_1 || "x";', {"SC_1": "A"}, "String", "Ax", id="computed"),
    pytest.param(
        "define operator f(x VD_1) returns string is x end operator; SC_r <- f(SC_2);",
        {"SC_2": "A"},
        "VD_1",
        "A",
        id="udo_parameter_on_value_domain",
    ),
    pytest.param(
        "define operator f(x string) returns string is x end operator; SC_r <- f(SC_1);",
        {"SC_1": "A"},
        "VD_1",
        "A",
        id="udo_parameter_of_basic_type",
    ),
    pytest.param(
        'define operator f(x VD_1 default "B") returns string is x end operator; SC_r <- f();',
        {},
        "VD_1",
        "B",
        id="udo_default_on_value_domain",
    ),
]


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize("script, scalar_values, type_, value", scalar_result_params)
def test_GH_1328_scalar_result(use_duckdb, script, scalar_values, type_, value):
    """A scalar result keeps the value domain that its operands share, and loses it when an
    operator computes a new value or the operands are not all on that value domain."""
    result = _run(script, scalar_values, use_duckdb)["SC_r"]
    assert result.to_dict()["type"] == type_
    assert result.value == value


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1328_scalar_into_component(use_duckdb):
    """A component that a scalar gives is defined on the value domain of the scalar, and one
    that a component and a scalar on the same value domain give keeps it."""
    result = _run(
        "DS_r <- DS_1 [calc Me_2 := SC_1, Me_3 := nvl(Me_1, SC_1)];", {"SC_1": "B"}, use_duckdb
    )["DS_r"]
    assert result.components["Me_2"].value_domain.name == "VD_1"
    assert result.components["Me_3"].value_domain.name == "VD_1"
    assert result.data.sort_values("Id_1")["Me_3"].tolist() == ["A", "B"]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1328_value_outside_value_domain(use_duckdb):
    """The value of an input scalar must be one of the values of its value domain."""
    with pytest.raises(InputValidationException, match="0-1-1-36"):
        _run("SC_r <- SC_1;", {"SC_1": "Z"}, use_duckdb)


downcast_error_params = [
    pytest.param("SC_r <- cast(SC_2, VD_1);", {"SC_2": "Q"}, "cast", id="cast"),
    pytest.param(
        "define operator f(x VD_1) returns string is x end operator; SC_r <- f(SC_2);",
        {"SC_2": "Q"},
        "f",
        id="udo_parameter",
    ),
    pytest.param(
        'define operator f(x VD_1 default "Q") returns string is x end operator; SC_r <- f();',
        {},
        "f",
        id="udo_default",
    ),
]


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize("script, scalar_values, op", downcast_error_params)
def test_GH_1328_downcast_outside_value_domain(use_duckdb, script, scalar_values, op):
    """A value of the basic type that a cast or an operator parameter puts into a value domain
    must be one of its values (downcast)."""
    with pytest.raises(RunTimeError, match=f"At op {op}: Cannot cast .* to Value Domain VD_1"):
        _run(script, scalar_values, use_duckdb)


def test_GH_1328_semantic_analysis():
    """The semantic analysis gives the scalar results their value domain."""
    result = semantic_analysis(
        script="SC_r <- SC_1; SC_s <- SC_1 || SC_2;",
        data_structures=DATA_STRUCTURES,
        value_domains=VALUE_DOMAINS,
    )
    assert result["SC_r"].to_dict()["type"] == "VD_1"
    assert result["SC_s"].to_dict()["type"] == "String"


def test_GH_1328_scalar_model():
    """A scalar on a value domain is written with its name, and two scalars on different
    value domains are not equal."""
    value_domain = VALUE_DOMAINS[0]
    scalar = Scalar("SC_1", String, "A", value_domain=value_domain)
    assert scalar.to_dict() == {"name": "SC_1", "type": "VD_1", "value": "A", "nullable": True}
    assert scalar != Scalar("SC_1", String, "A")
    assert scalar == Scalar("SC_1", String, "A", value_domain=value_domain)


# Issue #1349: a Time_Period or Date value is checked against a value domain as a value of its
# type, whatever form the value or the setlist is written in
VALUE_DOMAINS_1349 = [
    {"name": "VD_TP", "type": "Time_Period", "setlist": ["2024Q1", "2024Q2"]},
    {"name": "VD_DATE", "type": "Date", "setlist": ["2024-01-05"]},
]

DATA_STRUCTURES_1349 = {
    "scalars": [
        {"name": "SC_1", "type": "VD_TP"},
        {"name": "SC_2", "type": "Time_Period"},
        {"name": "SC_3", "type": "VD_DATE"},
    ]
}


def _run_1349(script, scalar_values, use_duckdb):
    return run(
        script=script,
        data_structures=DATA_STRUCTURES_1349,
        datapoints={},
        value_domains=VALUE_DOMAINS_1349,
        scalar_values=scalar_values,
        use_duckdb=use_duckdb,
    )["SC_r"]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_1(use_duckdb):
    """The scalar value 2024Q1 is the value 2024-Q1 of VD_TP; it raised 0-1-1-36."""
    result = _run_1349("SC_r <- SC_1;", {"SC_1": "2024Q1"}, use_duckdb)
    assert result.to_dict()["type"] == "VD_TP"
    assert result.value == "2024Q1"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_2(use_duckdb):
    """Control of case 1: the scalar value written 2024-Q1."""
    result = _run_1349("SC_r <- SC_1;", {"SC_1": "2024-Q1"}, use_duckdb)
    assert result.to_dict()["type"] == "VD_TP"
    assert result.value == "2024Q1"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_3(use_duckdb):
    """The Date value 2024-1-5 is the value 2024-01-05 of VD_DATE; it raised 0-1-1-36."""
    result = _run_1349("SC_r <- SC_3;", {"SC_3": "2024-1-5"}, use_duckdb)
    assert result.to_dict()["type"] == "VD_DATE"
    assert result.value == "2024-01-05"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_3b(use_duckdb):
    """A Date scalar on no value domain given as 2024-1-5 is the Date 2024-01-05, on both
    engines: pandas kept the text 2024-1-5, so the comparison was false."""
    result = run(
        script='SC_r <- SC_1 = cast("2024-01-05", date);',
        data_structures={"scalars": [{"name": "SC_1", "type": "Date"}]},
        datapoints={},
        scalar_values={"SC_1": "2024-1-5"},
        use_duckdb=use_duckdb,
    )["SC_r"]
    assert result.value is True


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_4(use_duckdb):
    """A Time_Period scalar 2024Q1 cast to VD_TP is a value of it; it raised 2-1-5-4."""
    result = _run_1349("SC_r <- cast(SC_2, VD_TP);", {"SC_2": "2024Q1"}, use_duckdb)
    assert result.to_dict()["type"] == "VD_TP"
    assert result.value == "2024Q1"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_5(use_duckdb):
    """The literal 2024Q1 cast to VD_TP is a value of it, also for the semantic analysis; it
    raised 1-1-5-6."""
    script = 'SC_r <- cast("2024Q1", VD_TP);'
    result = _run_1349(script, {}, use_duckdb)
    assert result.to_dict()["type"] == "VD_TP"
    assert result.value == "2024Q1"
    analysis = semantic_analysis(script, DATA_STRUCTURES_1349, VALUE_DOMAINS_1349)
    assert analysis["SC_r"].to_dict()["type"] == "VD_TP"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_6(use_duckdb):
    """A Time_Period scalar 2024Q1 passed to a parameter on VD_TP is a value of it; pandas
    raised 2-1-5-4."""
    script = (
        "define operator f (x VD_TP) returns string is cast(x, string) end operator;"
        " SC_r <- f(SC_2);"
    )
    result = _run_1349(script, {"SC_2": "2024Q1"}, use_duckdb)
    assert result.to_dict()["type"] == "String"
    assert result.value == "2024Q1"
