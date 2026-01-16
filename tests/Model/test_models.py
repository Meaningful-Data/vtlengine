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
    DataTypes.Null,
]


@pytest.mark.parametrize("type_param", type_params)
def test_str_representation(type_param):
    assert repr(type_param) == DataTypes.SCALAR_TYPES_CLASS_REVERSE[type_param]
    assert str(type_param) == DataTypes.SCALAR_TYPES_CLASS_REVERSE[type_param]


def test_component_serialization_uses_type():
    """Test that Component.to_dict() uses 'type' instead of 'data_type'"""
    comp = Component(
        name="test_comp", data_type=DataTypes.Integer, role=Role.MEASURE, nullable=True
    )
    comp_dict = comp.to_dict()

    assert "type" in comp_dict
    assert "data_type" not in comp_dict
    assert comp_dict["type"] == "Integer"
    assert comp_dict["name"] == "test_comp"
    assert comp_dict["role"] == "Measure"
    assert comp_dict["nullable"] is True


def test_component_from_json_supports_type():
    """Test that Component.from_json() accepts 'type' key"""
    json_data = {"name": "test_comp", "type": "String", "role": "Identifier", "nullable": False}
    comp = Component.from_json(json_data)

    assert comp.name == "test_comp"
    assert comp.data_type == DataTypes.String
    assert comp.role == Role.IDENTIFIER
    assert comp.nullable is False


def test_component_from_json_backward_compatibility():
    """Test that Component.from_json() still accepts 'data_type' key for backward compatibility"""
    json_data = {"name": "test_comp", "data_type": "Number", "role": "Measure", "nullable": True}
    comp = Component.from_json(json_data)

    assert comp.name == "test_comp"
    assert comp.data_type == DataTypes.Number
    assert comp.role == Role.MEASURE
    assert comp.nullable is True


def test_datacomponent_serialization_uses_type():
    """Test that DataComponent.to_dict() uses 'type' instead of 'data_type'"""
    dc = DataComponent(
        name="test_dc", data=None, data_type=DataTypes.Boolean, role=Role.ATTRIBUTE, nullable=True
    )
    dc_dict = dc.to_dict()

    assert "type" in dc_dict
    assert dc_dict["type"] == "Boolean"
    assert dc_dict["name"] == "test_dc"
    assert dc_dict["role"] == "Attribute"
    assert dc_dict["nullable"] is True


def test_datacomponent_from_json_supports_type():
    """Test that DataComponent.from_json() accepts 'type' key"""
    json_data = {"name": "test_dc", "type": "String", "role": "Measure", "nullable": False}
    dc = DataComponent.from_json(json_data)

    assert dc.name == "test_dc"
    assert dc.data_type == DataTypes.String
    assert dc.role == Role.MEASURE
    assert dc.nullable is False


def test_datacomponent_from_json_backward_compatibility():
    """Test that DataComponent.from_json() still accepts 'data_type' key for backward compatibility"""
    json_data = {"name": "test_dc", "data_type": "Integer", "role": "Identifier", "nullable": False}
    dc = DataComponent.from_json(json_data)

    assert dc.name == "test_dc"
    assert dc.data_type == DataTypes.Integer
    assert dc.role == Role.IDENTIFIER
    assert dc.nullable is False


def test_component_round_trip_serialization():
    """Test that Component can be serialized and deserialized correctly"""
    original = Component(
        name="round_trip", data_type=DataTypes.TimePeriod, role=Role.IDENTIFIER, nullable=False
    )

    # Serialize to dict
    comp_dict = original.to_dict()

    # Deserialize from dict
    restored = Component.from_json(comp_dict)

    # Verify they're equal
    assert original == restored
    assert original.name == restored.name
    assert original.data_type == restored.data_type
    assert original.role == restored.role
    assert original.nullable == restored.nullable
