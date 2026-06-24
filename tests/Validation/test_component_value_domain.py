from vtlengine.API._InternalApi import load_datasets
from vtlengine.DataTypes import Integer, String
from vtlengine.Model import Component, Role


def test_component_value_domain_roundtrip():
    c = Component(
        name="Me_1", data_type=String, role=Role.MEASURE, nullable=True, value_domain="CL_OBS"
    )
    assert c.value_domain == "CL_OBS"
    assert c.to_dict()["subset"] == "CL_OBS"
    assert Component.from_json(c.to_dict()).value_domain == "CL_OBS"
    assert c.copy().value_domain == "CL_OBS"


def test_component_value_domain_absent_by_default():
    c = Component(name="Id_1", data_type=Integer, role=Role.IDENTIFIER, nullable=False)
    assert c.value_domain is None
    assert "subset" not in c.to_dict()


def test_loader_populates_value_domain_from_subset():
    structures = {
        "datasets": [{"name": "DS_1", "structure": "S_1"}],
        "structures": [
            {
                "name": "S_1",
                "components": [
                    {
                        "name": "Id_1",
                        "data_type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "data_type": "String",
                        "role": "Measure",
                        "nullable": True,
                        "subset": "CL_OBS",
                    },
                ],
            }
        ],
    }
    datasets, _ = load_datasets(structures)
    assert datasets["DS_1"].components["Me_1"].value_domain == "CL_OBS"
    assert datasets["DS_1"].components["Id_1"].value_domain is None
