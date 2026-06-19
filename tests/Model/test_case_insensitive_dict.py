import copy
import pickle

import pytest

import vtlengine.DataTypes as DataTypes
from vtlengine.Model import CaseInsensitiveDict, Component, Dataset, Role


def test_case_insensitive_lookup():
    d: CaseInsensitiveDict[int] = CaseInsensitiveDict()
    d["Me_1"] = 10
    assert d["me_1"] == 10
    assert d["ME_1"] == 10
    assert "mE_1" in d


def test_preserves_first_seen_key():
    d: CaseInsensitiveDict[int] = CaseInsensitiveDict()
    d["Me_1"] = 1
    d["ME_1"] = 2  # updates value, keeps original display key
    assert list(d.keys()) == ["Me_1"]
    assert d["me_1"] == 2


def test_canonical_key():
    d: CaseInsensitiveDict[object] = CaseInsensitiveDict({"DS_1": object()})
    assert d.canonical_key("ds_1") == "DS_1"
    with pytest.raises(KeyError):
        d.canonical_key("missing")


def test_get_pop_setdefault():
    d: CaseInsensitiveDict[int] = CaseInsensitiveDict({"A": 1})
    assert d.get("a") == 1
    assert d.get("z", 99) == 99
    assert d.setdefault("a", 5) == 1
    assert d.pop("A") == 1
    assert "a" not in d


def test_to_dict_is_plain_with_original_keys():
    d: CaseInsensitiveDict[int] = CaseInsensitiveDict({"Me_1": 1})
    d["me_1"] = 2
    plain = d.to_dict()
    assert type(plain) is dict
    assert plain == {"Me_1": 2}


def test_deepcopy_and_pickle_roundtrip():
    d: CaseInsensitiveDict[list[int]] = CaseInsensitiveDict({"Me_1": [1, 2]})
    dc = copy.deepcopy(d)
    dc["ME_1"].append(3)
    assert d["me_1"] == [1, 2]  # independent
    assert dc["me_1"] == [1, 2, 3]
    rt = pickle.loads(pickle.dumps(d))  # noqa: S301
    assert rt["me_1"] == [1, 2]
    assert isinstance(rt, CaseInsensitiveDict)


def test_dataset_components_is_case_insensitive_dict():
    comps = {
        "Id_1": Component(
            name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
        ),
        "Me_1": Component(
            name="Me_1", data_type=DataTypes.Number, role=Role.MEASURE, nullable=True
        ),
    }
    ds = Dataset(name="DS_1", components=comps, data=None)
    assert isinstance(ds.components, CaseInsensitiveDict)
    assert "me_1" in ds.components
    assert ds.components["ME_1"].name == "Me_1"
    assert ds.get_component("me_1").name == "Me_1"
