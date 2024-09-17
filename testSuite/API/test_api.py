from pathlib import Path

import pytest

import DataTypes
from API.Api import load_vtl, load_value_domains, load_datasets
from DataTypes import String
from Model import ValueDomain, Dataset, Component, Role

# Path selection
base_path = Path(__file__).parent
filepath_VTL = base_path / "data" / "vtl"
filepath_ValueDomains = base_path / "data" / "ValueDomain"
filepath_sql = base_path / "data" / "sql"
filepath_json = base_path / "data" / "DataStructure" / "input"
filepath_csv = base_path / "data" / "DataSet" / "input"
filepath_out_json = base_path / "data" / "DataStructure" / "output"
filepath_out_csv = base_path / "data" / "DataSet" / "output"

input_vtl_params_OK = [
    (filepath_VTL / '1.vtl', 'DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;'),
    ('DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;', 'DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;')
]

input_vtl_error_params = [
    (filepath_VTL, 'Invalid vtl file. Must have .vtl extension'),
    (filepath_csv / 'DS_1.csv', 'Invalid vtl file. Must have .vtl extension'),
    (filepath_VTL / '2.vtl', 'Invalid vtl file. Input does not exist'),
    ({'DS': 'dataset'}, 'Invalid vtl file. Input is not a Path object'),
    (2, 'Invalid vtl file. Input is not a Path object')
]

input_vd_OK = [
    (filepath_ValueDomains / 'VD_1.json'),
    ({"name": "AnaCreditCountries", "setlist": ["AT", "BE", "CY"], "type": "String"})
]

input_vd_error_params = [
    (filepath_VTL / 'VD_1.json', 'Invalid vd file. Input does not exist'),
    (filepath_VTL / '1.vtl', 'Invalid vd file. Must have .json extension'),
    (filepath_json / 'DS_1.json', 'Invalid format for ValueDomain. Requires name, type and setlist.'),
    (2, 'Invalid vd file. Input is not a Path object'),
    ({"setlist": ["AT", "BE", "CY"], "type": "String"},
     'Invalid format for ValueDomain. Requires name, type and setlist.')
]

load_datasets_input_params_OK = [
    (filepath_json / 'DS_1.json'),
    ({"datasets": [{"name": "DS_1",
                    "DataStructure": [{"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False},
                                      {"name": "Id_2", "role": "Identifier", "type": "String", "nullable": False},
                                      {"name": "Me_1", "role": "Measure", "type": "Number", "nullable": True}]}]})
]

load_datasets_wrong_input_params = [
    (filepath_json / 'VD_1.json', 'Invalid datastructure. Input does not exist'),
    (filepath_csv / 'DS_1.csv', 'Invalid datastructure. Must have .json extension')
]


@pytest.mark.parametrize('input, expression', input_vtl_params_OK)
def test_load_input_vtl(input, expression):
    text = load_vtl(input)
    result = text
    assert result == expression


@pytest.mark.parametrize('input, error_message', input_vtl_error_params)
def test_load_wrong_inputs_vtl(input, error_message):
    with pytest.raises(Exception, match=error_message):
        load_vtl(input)


@pytest.mark.parametrize('input', input_vd_OK)
def test_load_input_vd(input):
    result = load_value_domains(input)
    reference = ValueDomain(name="AnaCreditCountries", setlist=["AT", "BE", "CY"], type=String)
    assert "AnaCreditCountries" in result
    assert result["AnaCreditCountries"] == reference


@pytest.mark.parametrize('input, error_message', input_vd_error_params)
def test_load_wrong_inputs_vd(input, error_message):
    with pytest.raises(Exception, match=error_message):
        load_value_domains(input)


@pytest.mark.parametrize('datastructure', load_datasets_input_params_OK)
def test_load_datastructures(datastructure):
    result = load_datasets(datastructure)
    reference = Dataset(name="DS_1", components={
        'Id_1': Component(name='Id_1', data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False),
        'Id_2': Component(name='Id_2', data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False),
        'Me_1': Component(name='Me_1', data_type=DataTypes.Number, role=Role.MEASURE, nullable=True)},
                        data=None)
    assert "DS_1" in result
    assert result["DS_1"] == reference


@pytest.mark.parametrize('input, error_message', load_datasets_wrong_input_params)
def test_load_wrong_inputs_datastructures(input, error_message):
    with pytest.raises(Exception, match=error_message):
        load_datasets(input)

