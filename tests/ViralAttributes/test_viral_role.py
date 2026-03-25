import pandas as pd

from vtlengine.DataTypes import Integer, Number, String
from vtlengine.Model import Component, Dataset, Role, Role_keys


class TestViralAttributeRole:
    def test_viral_attribute_in_role_enum(self):
        assert Role.VIRAL_ATTRIBUTE.value == "Viral Attribute"

    def test_viral_attribute_in_role_keys(self):
        assert "Viral Attribute" in Role_keys

    def test_component_with_viral_attribute_role(self):
        comp = Component(
            name="At_1",
            data_type=String,
            role=Role.VIRAL_ATTRIBUTE,
            nullable=True,
        )
        assert comp.role == Role.VIRAL_ATTRIBUTE

    def test_dataset_get_viral_attributes(self):
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", Number, Role.MEASURE, True),
                "At_1": Component("At_1", String, Role.VIRAL_ATTRIBUTE, True),
            },
            data=pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["A"]}),
        )
        viral_attrs = ds.get_viral_attributes()
        assert len(viral_attrs) == 1
        assert viral_attrs[0].name == "At_1"

    def test_dataset_get_viral_attributes_names(self):
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False),
                "At_1": Component("At_1", String, Role.VIRAL_ATTRIBUTE, True),
            },
            data=pd.DataFrame({"Id_1": [1], "At_1": ["A"]}),
        )
        assert ds.get_viral_attributes_names() == ["At_1"]

    def test_get_attributes_excludes_viral(self):
        """get_attributes() must NOT return VIRAL_ATTRIBUTE components."""
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False),
                "At_1": Component("At_1", String, Role.ATTRIBUTE, True),
                "At_2": Component("At_2", String, Role.VIRAL_ATTRIBUTE, True),
            },
            data=pd.DataFrame({"Id_1": [1], "At_1": ["A"], "At_2": ["B"]}),
        )
        attrs = ds.get_attributes()
        assert len(attrs) == 1
        assert attrs[0].name == "At_1"
        assert "At_2" not in ds.get_attributes_names()
