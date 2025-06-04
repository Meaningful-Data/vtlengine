from vtlengine.DataTypes import Integer, Number
from vtlengine.Model import Component, Dataset, Role
from vtlengine.Operators.Aggregation import Aggregation
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


def test_aggregation_generates_virtual_dataset_name():
    VirtualCounter.reset()
    ds = Dataset(
        name="DS_1",
        components={
            "Id_1": Component(name="Id_1", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
            "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
        },
        data=None,
    )
    result = Aggregation.validate(
        operand=ds, group_op=None, grouping_columns=None, having_data=None
    )
    assert result.name == "@VDS_1"
    assert result.name.startswith("@VDS_")


def test_aggregation_generates_virtual_dataset_name_2_ds():
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
    result_1 = Aggregation.validate(
        operand=ds_1, group_op=None, grouping_columns=None, having_data=None
    )
    result_2 = Aggregation.validate(
        operand=ds_2, group_op=None, grouping_columns=None, having_data=None
    )
    assert result_1.name == "@VDS_1"
    assert result_2.name == "@VDS_2"
    assert result_1.name.startswith("@VDS_")
    vc = VirtualCounter
    assert vc.dataset_count == 2
