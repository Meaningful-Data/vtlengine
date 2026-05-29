"""Viral attribute propagation through analytic (window) invocation."""

import pandas as pd

from vtlengine import run


def test_analytic_aggregate_viral_over_partition() -> None:
    """An aggregate-max viral rule reduces the attr over the partition (broadcast)."""
    script = """
        define viral propagation S (variable VAt_2) is
            aggregate max
        end viral propagation;
        DS_r <- sum(DS_1 over (partition by Id_1));
    """
    ds = {
        "name": "DS_1",
        "DataStructure": [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            {"name": "VAt_2", "type": "Integer", "role": "Viral Attribute", "nullable": True},
        ],
    }
    result = run(
        script=script,
        data_structures={"datasets": [ds]},
        datapoints={
            "DS_1": pd.DataFrame(
                {
                    "Id_1": [1, 1, 2],
                    "Id_2": [1, 2, 1],
                    "Me_1": [10.0, 20.0, 30.0],
                    "VAt_2": [3, 7, 5],
                }
            )
        },
    )
    data = result["DS_r"].data.sort_values(["Id_1", "Id_2"]).reset_index(drop=True)
    assert list(data["VAt_2"]) == [7, 7, 5]  # max over partition Id_1


def test_analytic_passthrough_no_rule() -> None:
    """With no propagation rule, the viral attr is passed through per-row (not dropped)."""
    script = "DS_r <- sum(DS_1 over (partition by Id_1));"
    ds = {
        "name": "DS_1",
        "DataStructure": [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
        ],
    }
    result = run(
        script=script,
        data_structures={"datasets": [ds]},
        datapoints={
            "DS_1": pd.DataFrame(
                {
                    "Id_1": [1, 1, 2],
                    "Id_2": [1, 2, 1],
                    "Me_1": [10.0, 20.0, 30.0],
                    "VAt_1": ["A", "B", "C"],
                }
            )
        },
    )
    data = result["DS_r"].data.sort_values(["Id_1", "Id_2"]).reset_index(drop=True)
    assert "VAt_1" in data.columns
    assert list(data["VAt_1"]) == ["A", "B", "C"]  # each row keeps its own value
