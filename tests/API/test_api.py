from pathlib import Path

import pandas as pd
import pytest

import vtlengine.DataTypes as DataTypes
from vtlengine.API import semantic_analysis, run
from vtlengine.API._InternalApi import (
    load_vtl,
    load_value_domains,
    load_datasets,
    load_datasets_with_data,
    load_external_routines,
)
from vtlengine.DataTypes import String
from vtlengine.Model import ValueDomain, Dataset, Component, Role, ExternalRoutine

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
    (filepath_VTL / "2.vtl", "DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;"),
    ("DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;", "DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;"),
]

input_vtl_error_params = [
    (filepath_VTL, "Invalid vtl file. Must have .vtl extension"),
    (filepath_csv / "DS_1.csv", "Invalid vtl file. Must have .vtl extension"),
    (filepath_VTL / "3.vtl", "Invalid vtl file. Input does not exist"),
    ({"DS": "dataset"}, "Invalid vtl file. Input is not a Path object"),
    (2, "Invalid vtl file. Input is not a Path object"),
]

input_vd_OK = [
    (filepath_ValueDomains / "VD_1.json"),
    ({"name": "AnaCreditCountries", "setlist": ["AT", "BE", "CY"], "type": "String"}),
]

input_vd_error_params = [
    (filepath_VTL / "VD_1.json", "Invalid vd file. Input does not exist"),
    (filepath_VTL / "1.vtl", "Invalid vd file. Must have .json extension"),
    (
        filepath_json / "DS_1.json",
        "Invalid format for ValueDomain. Requires name, type and setlist.",
    ),
    (2, "Invalid vd file. Input is not a Path object"),
    (
        {"setlist": ["AT", "BE", "CY"], "type": "String"},
        "Invalid format for ValueDomain. Requires name, type and setlist.",
    ),
]

load_datasets_input_params_OK = [
    (filepath_json / "DS_1.json"),
    (
        {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {
                            "name": "Id_1",
                            "role": "Identifier",
                            "type": "Integer",
                            "nullable": False,
                        },
                        {"name": "Id_2", "role": "Identifier", "type": "String", "nullable": False},
                        {"name": "Me_1", "role": "Measure", "type": "Number", "nullable": True},
                    ],
                }
            ]
        }
    ),
]

load_datasets_wrong_input_params = [
    (filepath_json / "VD_1.json", "Invalid datastructure. Input does not exist"),
    (filepath_csv / "DS_1.csv", "Invalid datastructure. Must have .json extension"),
]

load_datasets_with_data_without_dp_params_OK = [
    (
        filepath_json / "DS_1.json",
        None,
        (
            {
                "DS_1": Dataset(
                    name="DS_1",
                    components={
                        "Id_1": Component(
                            name="Id_1",
                            data_type=DataTypes.Integer,
                            role=Role.IDENTIFIER,
                            nullable=False,
                        ),
                        "Id_2": Component(
                            name="Id_2",
                            data_type=DataTypes.String,
                            role=Role.IDENTIFIER,
                            nullable=False,
                        ),
                        "Me_1": Component(
                            name="Me_1",
                            data_type=DataTypes.Number,
                            role=Role.MEASURE,
                            nullable=True,
                        ),
                    },
                    data=pd.DataFrame(columns=["Id_1", "Id_2", "Me_1"]),
                )
            },
            None,
        ),
    )
]

load_datasets_with_data_path_params_OK = [
    (
        filepath_json / "DS_1.json",
        filepath_csv / "DS_1.csv",
        (
            {
                "DS_1": Dataset(
                    name="DS_1",
                    components={
                        "Id_1": Component(
                            name="Id_1",
                            data_type=DataTypes.Integer,
                            role=Role.IDENTIFIER,
                            nullable=False,
                        ),
                        "Id_2": Component(
                            name="Id_2",
                            data_type=DataTypes.String,
                            role=Role.IDENTIFIER,
                            nullable=False,
                        ),
                        "Me_1": Component(
                            name="Me_1",
                            data_type=DataTypes.Number,
                            role=Role.MEASURE,
                            nullable=True,
                        ),
                    },
                    data=None,
                )
            },
            {"DS_1": filepath_csv / "DS_1.csv"},
        ),
    ),
    (
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        [filepath_csv / "DS_1.csv", filepath_csv / "DS_2.csv"],
        (
            {
                "DS_1": Dataset(
                    name="DS_1",
                    components={
                        "Id_1": Component(
                            name="Id_1",
                            data_type=DataTypes.Integer,
                            role=Role.IDENTIFIER,
                            nullable=False,
                        ),
                        "Id_2": Component(
                            name="Id_2",
                            data_type=DataTypes.String,
                            role=Role.IDENTIFIER,
                            nullable=False,
                        ),
                        "Me_1": Component(
                            name="Me_1",
                            data_type=DataTypes.Number,
                            role=Role.MEASURE,
                            nullable=True,
                        ),
                    },
                    data=None,
                ),
                "DS_2": Dataset(
                    name="DS_2",
                    components={
                        "Id_1": Component(
                            name="Id_1",
                            data_type=DataTypes.Integer,
                            role=Role.IDENTIFIER,
                            nullable=False,
                        ),
                        "Id_2": Component(
                            name="Id_2",
                            data_type=DataTypes.String,
                            role=Role.IDENTIFIER,
                            nullable=False,
                        ),
                        "Me_1": Component(
                            name="Me_1",
                            data_type=DataTypes.Number,
                            role=Role.MEASURE,
                            nullable=True,
                        ),
                    },
                    data=None,
                ),
            },
            {"DS_1": filepath_csv / "DS_1.csv", "DS_2": filepath_csv / "DS_2.csv"},
        ),
    ),
]

load_datasets_with_data_and_wrong_inputs = [
    (
        filepath_csv / "DS_1.csv",
        filepath_csv / "DS_1.csv",
        "Invalid datastructure. Must have .json extension",
    ),
    (filepath_json / "DS_1.json", filepath_json / "DS_2.json", "Not found dataset DS_2.json"),
    (2, 2, "Invalid datastructure. Input must be a dict or Path object"),
]

ext_params_OK = [(filepath_sql / "1.sql")]

ext_params_wrong = [
    (filepath_json / "DS_1.json", "Input must be a sql file"),
    (5, "Input invalid. Input must be a sql file."),
    (filepath_sql / "2.sql", "Input invalid. Input does not exist"),
]

params_semantic = [
    (
        filepath_VTL / "1.vtl",
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        filepath_ValueDomains / "VD_1.json",
        filepath_sql / "1.sql",
    )
]

params_run = [
    (
        filepath_VTL / "2.vtl",
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        [filepath_csv / "DS_1.csv", filepath_csv / "DS_2.csv"],
        filepath_ValueDomains / "VD_1.json",
        filepath_sql / "1.sql",
    )
]


@pytest.mark.parametrize("input", ext_params_OK)
def test_load_external_routine(input):
    result = load_external_routines(input)
    reference = {
        "1": ExternalRoutine(
            dataset_names=["BNFCRS_TRNSFRS", "BNFCRS_TRNSFRS_CMMN_INSTRMNTS_4"],
            query="SELECT\n    date(DT_RFRNC) as DT_RFRNC,\n    PRSPCTV_ID,\n    INSTRMNT_UNQ_ID,\n    BNFCRS_CNTRPRTY_ID,\n    TRNSFR_CNTRPRTY_ID,\n    BNFCR_ID,\n    TRNSFR_ID\nFROM\n    BNFCRS_TRNSFRS\nWHERE\n    INSTRMNT_UNQ_ID NOT IN(\n\t\tSELECT\n\t\t\tINSTRMNT_UNQ_ID\n\t\tFROM\n\t\t\tBNFCRS_TRNSFRS_CMMN_INSTRMNTS_4);\n",
            name="1",
        )
    }

    assert result == reference


@pytest.mark.parametrize("input, error_message", ext_params_wrong)
def test_load_external_routine_with_wrong_params(input, error_message):
    with pytest.raises(Exception, match=error_message):
        load_external_routines(input)


@pytest.mark.parametrize("input, expression", input_vtl_params_OK)
def test_load_input_vtl(input, expression):
    text = load_vtl(input)
    result = text
    assert result == expression


@pytest.mark.parametrize("input, error_message", input_vtl_error_params)
def test_load_wrong_inputs_vtl(input, error_message):
    with pytest.raises(Exception, match=error_message):
        load_vtl(input)


@pytest.mark.parametrize("input", input_vd_OK)
def test_load_input_vd(input):
    result = load_value_domains(input)
    reference = ValueDomain(name="AnaCreditCountries", setlist=["AT", "BE", "CY"], type=String)
    assert "AnaCreditCountries" in result
    assert result["AnaCreditCountries"] == reference


@pytest.mark.parametrize("input, error_message", input_vd_error_params)
def test_load_wrong_inputs_vd(input, error_message):
    with pytest.raises(Exception, match=error_message):
        load_value_domains(input)


@pytest.mark.parametrize("datastructure", load_datasets_input_params_OK)
def test_load_datastructures(datastructure):
    result = load_datasets(datastructure)
    reference = Dataset(
        name="DS_1",
        components={
            "Id_1": Component(
                name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
            ),
            "Id_2": Component(
                name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
            ),
            "Me_1": Component(
                name="Me_1", data_type=DataTypes.Number, role=Role.MEASURE, nullable=True
            ),
        },
        data=None,
    )
    assert "DS_1" in result
    assert result["DS_1"] == reference


@pytest.mark.parametrize("input, error_message", load_datasets_wrong_input_params)
def test_load_wrong_inputs_datastructures(input, error_message):
    with pytest.raises(Exception, match=error_message):
        load_datasets(input)


@pytest.mark.parametrize("ds_r, dp, reference", load_datasets_with_data_without_dp_params_OK)
def test_load_datasets_with_data_without_dp(ds_r, dp, reference):
    result = load_datasets_with_data(data_structures=ds_r, datapoints=dp)
    assert result == reference


@pytest.mark.parametrize("ds_r, dp, reference", load_datasets_with_data_path_params_OK)
def test_load_datasets_with_data_path(ds_r, dp, reference):
    result = load_datasets_with_data(data_structures=ds_r, datapoints=dp)
    assert result == reference


@pytest.mark.parametrize("ds_r, dp, error_message", load_datasets_with_data_and_wrong_inputs)
def test_load_datasets_with_wrong_inputs(ds_r, dp, error_message):
    with pytest.raises(Exception, match=error_message):
        load_datasets_with_data(ds_r, dp)


@pytest.mark.parametrize(
    "script, data_structures, value_domains, external_routines", params_semantic
)
def test_semantic(script, data_structures, value_domains, external_routines):
    result = semantic_analysis(script, data_structures, value_domains, external_routines)
    reference = {
        "DS_r": Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(
                    name="Me_1", data_type=DataTypes.Number, role=Role.MEASURE, nullable=True
                ),
            },
            data=None,
        )
    }

    assert result == reference


@pytest.mark.parametrize(
    "script, data_structures, datapoints, value_domains, external_routines", params_run
)
def test_run(script, data_structures, datapoints, value_domains, external_routines):
    result = run(script, data_structures, datapoints, value_domains, external_routines)
    reference = {
        "DS_r": Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(
                    name="Me_1", data_type=DataTypes.Number, role=Role.MEASURE, nullable=True
                ),
            },
            data=pd.DataFrame(
                columns=["Id_1", "Id_2", "Me_1"], index=[0, 1], data=[(1, "A", 2), (1, "B", 4)]
            ),
        ),
        "DS_r2": Dataset(
            name="DS_r2",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(
                    name="Me_1", data_type=DataTypes.Number, role=Role.MEASURE, nullable=True
                ),
            },
            data=pd.DataFrame(
                columns=["Id_1", "Id_2", "Me_1"], index=[0, 1], data=[(1, "A", 3), (1, "B", 6)]
            ),
        ),
    }

    assert result == reference


@pytest.mark.parametrize(
    "script, data_structures, datapoints, value_domains, external_routines", params_run
)
def test_run_only_persistent(script, data_structures, datapoints, value_domains, external_routines):
    result = run(
        script,
        data_structures,
        datapoints,
        value_domains,
        external_routines,
        return_only_persistent=True,
    )
    reference = {
        "DS_r2": Dataset(
            name="DS_r2",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(
                    name="Me_1", data_type=DataTypes.Number, role=Role.MEASURE, nullable=True
                ),
            },
            data=pd.DataFrame(
                columns=["Id_1", "Id_2", "Me_1"], index=[0, 1], data=[(1, "A", 3), (1, "B", 6)]
            ),
        )
    }

    assert result == reference
