"""Viral attribute propagation (merge) through join operators."""

import pandas as pd

from vtlengine import run

CONF_RULE = """
    define viral propagation CONF (variable VAt_1) is
        when "C" then "C";
        when "N" then "N";
        else "F"
    end viral propagation;
"""

SMAX_RULE = """
    define viral propagation S (variable VAt_1) is
        aggregate max
    end viral propagation;
"""


def _ds(name: str) -> dict:
    return {
        "name": name,
        "DataStructure": [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
        ],
    }


def test_inner_join_merges_viral_into_single_column() -> None:
    result = run(
        script=CONF_RULE + "DS_r <- inner_join(DS_1, DS_2);",
        data_structures={"datasets": [_ds("DS_1"), _ds("DS_2")]},
        datapoints={
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["C", "N"]}),
            "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0], "VAt_1": ["N", "F"]}),
        },
    )
    ds_r = result["DS_r"]
    # exactly ONE viral column named VAt_1 (no DS_1#VAt_1 / DS_2#VAt_1)
    assert [c for c in ds_r.components if "VAt_1" in c] == ["VAt_1"]
    assert list(ds_r.data.sort_values("Id_1")["VAt_1"]) == ["C", "N"]  # C+N->C ; N+F->N


def test_inner_join_viral_no_rule_is_null() -> None:
    result = run(
        script="DS_r <- inner_join(DS_1, DS_2);",
        data_structures={"datasets": [_ds("DS_1"), _ds("DS_2")]},
        datapoints={
            "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["C"]}),
            "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "VAt_1": ["N"]}),
        },
    )
    ds_r = result["DS_r"]
    assert [c for c in ds_r.components if "VAt_1" in c] == ["VAt_1"]
    assert pd.isna(ds_r.data["VAt_1"].iloc[0])


def test_left_join_viral_merge_aggregate() -> None:
    result = run(
        script=SMAX_RULE + "DS_r <- left_join(DS_1, DS_2);",
        data_structures={"datasets": [_ds("DS_1"), _ds("DS_2")]},
        datapoints={
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["A", "C"]}),
            "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "VAt_1": ["B"]}),
        },
    )
    ds_r = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
    # Id_1=1: GREATEST('A','B')='B'; Id_1=2: only left 'C' (right NULL, ignored) -> 'C'
    assert list(ds_r["VAt_1"]) == ["B", "C"]
