import pandas as pd

from vtlengine import run, semantic_analysis
from vtlengine.DataTypes import Integer, String
from vtlengine.Model import ValueDomain
from vtlengine.Operators.Validation import resolve_error_types

DS_STRUCT = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}

DPR_SCRIPT = """
define datapoint ruleset dpr1 ( variable Me_1 ) is
    when Me_1 > 0 then Me_1 < 100 errorcode "bad" errorlevel 5
end datapoint ruleset;
DS_r <- check_datapoint ( DS_1, dpr1 all );
"""

CHECK_SCRIPT = """
DS_r <- check ( DS_1 > 0 errorcode "x" errorlevel 1 invalid );
"""


def test_resolve_error_types_defaults():
    assert resolve_error_types(None) == (String, Integer)
    assert resolve_error_types({}) == (String, Integer)


def test_resolve_error_types_overrides():
    vds = {
        "errorcode_vd": ValueDomain(name="errorcode_vd", type=Integer, setlist=[1, 2]),
        "errorlevel_vd": ValueDomain(name="errorlevel_vd", type=String, setlist=["a", "b"]),
    }
    assert resolve_error_types(vds) == (Integer, String)


def test_check_datapoint_default_errorlevel_is_integer():
    res = semantic_analysis(script=DPR_SCRIPT, data_structures=DS_STRUCT)
    comps = res["DS_r"].components
    assert comps["errorlevel"].data_type == Integer
    assert comps["errorcode"].data_type == String


def test_check_datapoint_errorlevel_vd_override():
    vds = [{"name": "errorlevel_vd", "setlist": ["high", "low"], "type": "String"}]
    res = semantic_analysis(script=DPR_SCRIPT, data_structures=DS_STRUCT, value_domains=vds)
    assert res["DS_r"].components["errorlevel"].data_type == String


def test_check_datapoint_errorcode_vd_override():
    vds = [{"name": "errorcode_vd", "setlist": [1, 2], "type": "Integer"}]
    res = semantic_analysis(script=DPR_SCRIPT, data_structures=DS_STRUCT, value_domains=vds)
    assert res["DS_r"].components["errorcode"].data_type == Integer


def test_check_operator_default_types():
    res = semantic_analysis(script=CHECK_SCRIPT, data_structures=DS_STRUCT)
    comps = res["DS_r"].components
    assert comps["errorcode"].data_type == String
    assert comps["errorlevel"].data_type == Integer


def test_check_operator_value_domains_override():
    vds = [
        {"name": "errorcode_vd", "setlist": [1, 2], "type": "Integer"},
        {"name": "errorlevel_vd", "setlist": ["high", "low"], "type": "String"},
    ]
    res = semantic_analysis(script=CHECK_SCRIPT, data_structures=DS_STRUCT, value_domains=vds)
    comps = res["DS_r"].components
    assert comps["errorcode"].data_type == Integer
    assert comps["errorlevel"].data_type == String


def test_check_datapoint_run_errorlevel_integer():
    data = {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 200.0]})}
    res = run(script=DPR_SCRIPT, data_structures=DS_STRUCT, datapoints=data)
    assert res["DS_r"].components["errorlevel"].data_type == Integer


def test_check_datapoint_run_errorlevel_vd_override():
    data = {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 200.0]})}
    vds = [{"name": "errorlevel_vd", "setlist": ["high", "low"], "type": "String"}]
    res = run(script=DPR_SCRIPT, data_structures=DS_STRUCT, datapoints=data, value_domains=vds)
    assert res["DS_r"].components["errorlevel"].data_type == String
