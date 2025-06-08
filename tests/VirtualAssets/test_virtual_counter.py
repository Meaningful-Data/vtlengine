from pathlib import Path
from unittest.mock import patch

import pandas as pd

from vtlengine import run
from vtlengine.DataTypes import Integer, Number
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar
from vtlengine.Operators import Unary
from vtlengine.Operators.Analytic import Analytic
from vtlengine.Operators.Conditional import Nvl
from vtlengine.Utils.__Virtual_Assets import VirtualCounter

base_path = Path(__file__).parent
filepath_VTL = base_path / "data" / "vtl"
filepath_json = base_path / "data" / "DataStructure" / "input"
filepath_csv = base_path / "data" / "DataSet" / "input"


def test_analytic_generates_virtual_dataset_name():
    VirtualCounter.reset()
    ds = Dataset(
        name="DS_1",
        components={
            "Id_1": Component(name="Id_1", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
            "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
        },
        data=None,
    )
    result = Analytic.validate(
        operand=ds, partitioning=[], ordering=None, window=None, params=None, component_name=None
    )
    assert result.name == "@VDS_1"
    assert result.name.startswith("@VDS_")


def test_analytic_generates_virtual_dataset_name_2_ds():
    VirtualCounter.reset()
    ds_1 = Dataset(
        name="DS_1",
        components={
            "Id_1": Component(name="Id_1", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
            "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
        },
        data=None,
    )
    ds_2 = Dataset(
        name="DS_2",
        components={
            "Id_2": Component(name="Id_2", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
            "Me_2": Component(name="Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
        },
        data=None,
    )
    result_1 = Analytic.validate(
        operand=ds_1, partitioning=[], ordering=None, window=None, params=None, component_name=None
    )
    result_2 = Analytic.validate(
        operand=ds_2, partitioning=[], ordering=None, window=None, params=None, component_name=None
    )
    assert result_1.name == "@VDS_1"
    assert result_2.name == "@VDS_2"
    assert result_1.name.startswith("@VDS_")
    vc = VirtualCounter
    assert vc.dataset_count == 2


def test_binary_generates_virtual_dataset_name():
    VirtualCounter.reset()
    ds_left = Dataset(
        name="DS_1",
        components={
            "Id_1": Component("Id_1", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
            "Me_1": Component("Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            "Me_2": Component("Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
        },
        data=None,
    )
    scalar_right = Scalar(name="test", value=0, data_type=Number)

    result = Nvl.validate(ds_left, scalar_right)
    assert result.name == "@VDS_1"
    assert result.name.startswith("@VDS_")
    assert VirtualCounter.dataset_count == 1
    assert VirtualCounter.component_count == 1


def test_binary_generates_virtual_component_name():
    VirtualCounter.reset()
    left_comp = DataComponent(
        name="Me_1",
        data=None,
        data_type=Number,
        role=Role.MEASURE,
        nullable=True,
    )
    right_scalar = Scalar(name="test", value=0, data_type=Number)

    result = Nvl.validate(left_comp, right_scalar)
    assert result.name == "@VDC_1"
    assert result.role == Role.MEASURE
    assert VirtualCounter.dataset_count == 1
    assert VirtualCounter.component_count == 1


def test_unary_generates_virtual_dataset_name():
    VirtualCounter.reset()
    ds_left = Dataset(
        name="DS_1",
        components={
            "Id_1": Component("Id_1", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
            "Me_1": Component("Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            "Me_2": Component("Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
        },
        data=None,
    )

    result = Unary.validate(ds_left)

    assert result.name == "@VDS_1"
    assert result.name.startswith("@VDS_")
    assert VirtualCounter.dataset_count == 1
    assert VirtualCounter.component_count == 0


def test_unary_generates_virtual_component_name():
    VirtualCounter.reset()

    left_comp = DataComponent(
        name="Me_1",
        data=None,
        data_type=Number,
        role=Role.MEASURE,
        nullable=True,
    )
    result = Unary.validate(left_comp)

    assert result.name == "@VDC_1"
    assert result.role == Role.MEASURE
    assert VirtualCounter.dataset_count == 0
    assert VirtualCounter.component_count == 1


def test_components_generates_virtual_component():
    VirtualCounter.reset()
    assert VirtualCounter.component_count == 0
    operand = DataComponent(
        name="Me_1",
        data_type=Integer,
        data=None,
        role=Role.MEASURE,
        nullable=True,
    )
    result = Analytic.component_validation(operand)
    assert result.name == "@VDC_1"
    assert VirtualCounter.component_count == 1


def test_multiple_components_increments_counter():
    VirtualCounter.reset()
    assert VirtualCounter.component_count == 0
    operand = DataComponent(
        name="Me_1",
        data_type=Integer,
        data=None,
        role=Role.MEASURE,
        nullable=True,
    )
    results = []
    expected_names = []

    for i in range(1, 6):
        result = Analytic.component_validation(operand)
        results.append(result)
        expected_names.append(f"@VDC_{i}")
        assert result.name == f"@VDC_{i}"
    all_names = [comp.name for comp in results]
    assert all_names == expected_names
    assert VirtualCounter.component_count == 5


def test_virtual_counter_with_run():
    VirtualCounter.reset()
    script = """
           DS_r := DS_1 * 10;
           DS_r := DS_1 [ calc Me_1:= Me_1 * 2 ];
           DS_r := inner_join ( DS_1  filter Id_2="B" calc Me_2:=Me_1);
           DS_r := DS_1[calc Me_3 := daytomonth(Me_2)];
       """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                    {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Id_2": ["A", "B", "C"], "Me_1": [10, 20, 30]})

    datapoints = {"DS_1": data_df}
    call_vds = []
    call_vdc = []

    def mock_new_ds_name():
        ds = f"@VDS_{len(call_vds) + 1}"
        call_vds.append(ds)
        return ds

    def mock_new_dc_name():
        dc = f"@VDC_{len(call_vdc) + 1}"
        call_vdc.append(dc)
        return dc

    with patch(
        "vtlengine.Utils.__Virtual_Assets.VirtualCounter._new_ds_name", side_effect=mock_new_ds_name
    ):
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
    with patch(
        "vtlengine.Utils.__Virtual_Assets.VirtualCounter._new_dc_name", side_effect=mock_new_dc_name
    ):
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
    assert "DS_r" in result
    assert len(call_vds) == 6
    assert len(call_vdc) == 1
    assert VirtualCounter.dataset_count == 0
    assert VirtualCounter.component_count == 0


def test_virtual_counter_aggregate():
    VirtualCounter.reset()
    script = """
        DS_r := DS_1[aggr Me_2 := sum(Me_1) group by Id_2];
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                    {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Id_2": ["A", "B", "C"], "Me_1": [10, 20, 30]})

    datapoints = {"DS_1": data_df}
    call_vds = []

    def mock_new_ds_name():
        ds = f"@VDS_{len(call_vds) + 1}"
        call_vds.append(ds)
        return ds

    with patch(
        "vtlengine.Utils.__Virtual_Assets.VirtualCounter._new_ds_name", side_effect=mock_new_ds_name
    ):
        run(script=script, data_structures=data_structures, datapoints=datapoints)
    assert len(call_vds) == 1
    assert set(call_vds) == {"@VDS_1"}
    assert VirtualCounter.dataset_count == 0
    assert VirtualCounter.component_count == 0


def test_virtual_counter_analytic():
    VirtualCounter.reset()
    script = """
        DS_r := first_value ( DS_1 over ( partition by Id_1, Id_2));
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                    {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Id_2": ["A", "B", "C"], "Me_1": [10, 20, 30]})

    datapoints = {"DS_1": data_df}
    call_vds = []

    def mock_new_ds_name():
        ds = f"@VDS_{len(call_vds) + 1}"
        call_vds.append(ds)
        return ds

    with patch(
        "vtlengine.Utils.__Virtual_Assets.VirtualCounter._new_ds_name", side_effect=mock_new_ds_name
    ):
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
    assert len(result["DS_r"].data) == 3
    assert len(call_vds) == 1
    assert VirtualCounter.dataset_count == 0
    assert VirtualCounter.component_count == 0


def test_virtual_counter_run_with_udo():
    VirtualCounter.reset()
    script = filepath_VTL / "UDO.vtl"
    data_structures = [filepath_json / "DS_1.json", filepath_json / "DS_2.json"]
    datapoints = {
        "DS_1": pd.read_csv(filepath_csv / "DS_1.csv"),
        "DS_2": pd.read_csv(filepath_csv / "DS_2.csv"),
    }
    call_vds = []
    call_vdc = []

    def mock_new_ds_name():
        ds = f"@VDS_{len(call_vds) + 1}"
        call_vds.append(ds)
        return ds

    def mock_new_dc_name():
        dc = f"@VDC_{len(call_vdc) + 1}"
        call_vdc.append(dc)
        return dc

    with patch(
        "vtlengine.Utils.__Virtual_Assets.VirtualCounter._new_ds_name", side_effect=mock_new_ds_name
    ):
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
    with patch(
        "vtlengine.Utils.__Virtual_Assets.VirtualCounter._new_dc_name", side_effect=mock_new_dc_name
    ):
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)

    assert len(call_vds) == 2
    assert len(call_vdc) == 0
    assert result["DS_r"].data["Me_1"].tolist() == [15, 6, 9, 12, 30, 12, 18, 24]
    assert VirtualCounter.dataset_count == 0
    assert VirtualCounter.component_count == 0
