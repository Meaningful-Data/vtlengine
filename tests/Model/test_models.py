from typing import Any, Dict, List, Tuple

import pytest

import vtlengine.DataTypes as DataTypes
from vtlengine.Model import (
    Component,
    DataComponent,
    Dataset,
    ExternalRoutine,
    Role,
    Scalar,
    ScalarSet,
    ValueDomain,
)

ds_components = {
    "id": Component(name="id", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False),
    "value": Component(name="me", data_type=DataTypes.String, role=Role.MEASURE, nullable=True),
}

ds_components2 = {
    "id": Component(name="id", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False),
    "value": Component(name="value", data_type=DataTypes.Number, role=Role.MEASURE, nullable=True),
    "attr": Component(name="attr", data_type=DataTypes.String, role=Role.ATTRIBUTE, nullable=True),
}
ds_components3 = {
    "key": Component(name="key", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False),
    "flag": Component(name="flag", data_type=DataTypes.Boolean, role=Role.MEASURE, nullable=True),
    "desc": Component(name="desc", data_type=DataTypes.String, role=Role.ATTRIBUTE, nullable=True),
}

params: List[Tuple[type, Dict[str, Any]]] = [
    (Scalar, {"name": "sc_int", "data_type": DataTypes.Integer, "value": 1}),
    (Scalar, {"name": "sc_str", "data_type": DataTypes.String, "value": "abc"}),
    (Scalar, {"name": "sc_bool", "data_type": DataTypes.Boolean, "value": True}),
    (Scalar, {"name": "sc_num", "data_type": DataTypes.Number, "value": 3.14}),
    (
        Component,
        {
            "name": "id_int",
            "data_type": DataTypes.Integer,
            "role": Role.IDENTIFIER,
            "nullable": False,
        },
    ),
    (
        Component,
        {"name": "me_num", "data_type": DataTypes.Number, "role": Role.MEASURE, "nullable": True},
    ),
    (
        Component,
        {
            "name": "attr_str",
            "data_type": DataTypes.String,
            "role": Role.ATTRIBUTE,
            "nullable": True,
        },
    ),
    (
        Component,
        {"name": "", "data_type": DataTypes.Boolean, "role": Role.MEASURE, "nullable": True},
    ),
    (
        DataComponent,
        {
            "name": "data_comp",
            "data": None,
            "data_type": DataTypes.Number,
            "role": Role.MEASURE,
            "nullable": True,
        },
    ),
    (
        DataComponent,
        {
            "name": "id_dc",
            "data": [1, 2, 3],
            "data_type": DataTypes.Integer,
            "role": Role.IDENTIFIER,
            "nullable": False,
        },
    ),
    (
        DataComponent,
        {
            "name": "attr_dc",
            "data": ["A", "B"],
            "data_type": DataTypes.String,
            "role": Role.ATTRIBUTE,
            "nullable": True,
        },
    ),
    (
        DataComponent,
        {
            "name": "flag_dc",
            "data": [True, False],
            "data_type": DataTypes.Boolean,
            "role": Role.MEASURE,
            "nullable": True,
        },
    ),
    (Dataset, {"name": "ds", "components": ds_components}),
    (Dataset, {"name": "ds2", "components": ds_components2}),
    (Dataset, {"name": "ds3", "components": ds_components3, "persistent": True}),
    (ScalarSet, {"data_type": DataTypes.Integer, "values": [1, 2, 3]}),
    (ScalarSet, {"data_type": DataTypes.String, "values": ["A", "B"]}),
    (ScalarSet, {"data_type": DataTypes.Boolean, "values": [True, False]}),
    (ScalarSet, {"data_type": DataTypes.Number, "values": [0.1, 2.5, 3.0]}),
    (ValueDomain, {"name": "status", "type": DataTypes.String, "setlist": ["OPEN", "CLOSED"]}),
    (ValueDomain, {"name": "priority", "type": DataTypes.Integer, "setlist": [1, 2, 3]}),
    (ValueDomain, {"name": "flags", "type": DataTypes.Boolean, "setlist": [True, False]}),
    (
        ExternalRoutine,
        {
            "dataset_names": ["t1", "t2"],
            "query": "SELECT * FROM t1 JOIN t2 ON 1=1",
            "name": "my_routine",
        },
    ),
    (
        ExternalRoutine,
        {
            "dataset_names": ["sales"],
            "query": "SELECT * FROM sales WHERE amount > 0",
            "name": "r_sales",
        },
    ),
    (
        ExternalRoutine,
        {
            "dataset_names": ["a", "b"],
            "query": "SELECT a.id, b.val FROM a JOIN b ON a.id = b.id",
            "name": "r_join",
        },
    ),
]


@pytest.mark.parametrize("class_type, init_args", params)
def test_datamodel_initialization(class_type, init_args):
    instance = class_type(**init_args)
    for k, v in init_args.items():
        assert hasattr(instance, k), f"{class_type.__name__} doesnt have the attr {k}"
        assert getattr(instance, k) == v


type_params = [
    DataTypes.Integer,
    DataTypes.String,
    DataTypes.Boolean,
    DataTypes.Number,
    DataTypes.TimeInterval,
    DataTypes.TimePeriod,
    DataTypes.Date,
    DataTypes.Duration,
]


@pytest.mark.parametrize("type_param", type_params)
def test_str_representation(type_param):
    assert repr(type_param) == DataTypes.SCALAR_TYPES_CLASS_REVERSE[type_param]
    assert str(type_param) == DataTypes.SCALAR_TYPES_CLASS_REVERSE[type_param]
