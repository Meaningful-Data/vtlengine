
from vtlengine import semantic_analysis
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
