import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from pysdmx.model import (
    Transformation,
    TransformationScheme,
)

import vtlengine.DataTypes as DataTypes
from tests.Helper import _use_duckdb_backend, engine_params
from vtlengine.API import (
    create_ast,
    prettify,
    run,
    semantic_analysis,
    validate_dataset,
    validate_external_routine,
    validate_value_domain,
)
from vtlengine.API._InternalApi import (
    _check_script,
    _validate_json,
    load_datasets,
    load_datasets_with_data,
    load_external_routines,
    load_value_domains,
    load_vtl,
)
from vtlengine.DataTypes import Integer, Null, String
from vtlengine.Exceptions import DataLoadError, InputValidationException, SemanticError
from vtlengine.Model import Component, Dataset, ExternalRoutine, Role, Scalar, ValueDomain
from vtlengine.Utils._recursion import MIN_RECURSION_LIMIT, recursion_headroom

# Path selection
base_path = Path(__file__).parent
filepath_VTL = base_path / "data" / "vtl"
filepath_ValueDomains = base_path / "data" / "ValueDomain"
filepath_sql = base_path / "data" / "sql"
filepath_json = base_path / "data" / "DataStructure" / "input"
filepath_csv = base_path / "data" / "DataSet" / "input"
filepath_out_json = base_path / "data" / "DataStructure" / "output"
filepath_out_csv = base_path / "data" / "DataSet" / "output"
filepath_sdmx_input = base_path / "data" / "SDMX" / "input"
filepath_sdmx_output = base_path / "data" / "SDMX" / "output"


input_vtl_params_OK = [
    (filepath_VTL / "2.vtl", "DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;"),
    (
        "DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;",
        "DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;",
    ),
]

input_vtl_error_params = [
    (filepath_VTL, "0-1-1-3"),
    (filepath_csv / "DS_1.csv", "0-1-1-3"),
    (filepath_VTL / "3.vtl", "0-3-1-1"),
    ({"DS": "dataset"}, "0-1-1-2"),
    (2, "0-1-1-2"),
]

input_vd_OK = [
    (filepath_ValueDomains / "VD_1.json"),
    ({"name": "AnaCreditCountries", "setlist": ["AT", "BE", "CY"], "type": "String"}),
]

input_vd_error_params = [
    (filepath_VTL / "VD_1.json", "0-3-1-1"),
    (filepath_VTL / "1.vtl", "0-1-1-3"),
    (
        filepath_json / "DS_1.json",
        "0-2-1-1",
    ),
    (2, "0-1-1-2"),
    (
        {"setlist": ["AT", "BE", "CY"], "type": "String"},
        "0-2-1-1",
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
                        {
                            "name": "Id_2",
                            "role": "Identifier",
                            "type": "String",
                            "nullable": False,
                        },
                        {
                            "name": "Me_1",
                            "role": "Measure",
                            "type": "Number",
                            "nullable": True,
                        },
                    ],
                }
            ],
            "scalars": [
                {
                    "name": "sc_1",
                    "type": "Integer",
                },
            ],
        }
    ),
]

load_datasets_wrong_input_params = [
    (filepath_json / "VD_1.json", "0-3-1-1"),
    (filepath_csv / "DS_1.csv", "0-1-1-3"),
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
            {"sc_1": Scalar(name="sc_1", data_type=DataTypes.Integer, value=None)},
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
                    data=pd.DataFrame(columns=["Id_1", "Id_2", "Me_1"]),
                )
            },
            {"sc_1": Scalar(name="sc_1", data_type=Integer, value=None)},
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
                    data=pd.DataFrame(columns=["Id_1", "Id_2", "Me_1"]),
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
                    data=pd.DataFrame(columns=["Id_1", "Id_2", "Me_1"]),
                ),
            },
            {"sc_1": Scalar(name="sc_1", data_type=Integer, value=None)},
            {"DS_1": filepath_csv / "DS_1.csv", "DS_2": filepath_csv / "DS_2.csv"},
        ),
    ),
    (
        filepath_json / "DS_1.json",
        {"DS_1": filepath_csv / "custom_name_DS_1.csv"},
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
            {"sc_1": Scalar(name="sc_1", data_type=Integer, value=None)},
            {"DS_1": filepath_csv / "custom_name_DS_1.csv"},
        ),
    ),
]

load_datasets_with_data_and_wrong_inputs = [
    (
        filepath_csv / "DS_1.csv",
        filepath_csv / "DS_1.csv",
        "0-1-1-3",
    ),
    (
        filepath_json / "DS_1.json",
        filepath_json / "DS_2.json",
        "Not found dataset DS_2.json",
    ),
    (2, 2, "0-1-1-2"),
]

ext_params_OK = [(filepath_sql / "1.json")]

ext_params_wrong = [
    (
        filepath_json / "DS_1.json",
        "0-2-1-1",
    ),
    (5, "0-1-1-2"),
    (filepath_sql / "6.sql", "0-3-1-1"),
]

params_semantic = [
    (
        filepath_VTL / "1.vtl",
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        filepath_ValueDomains / "VD_1.json",
        filepath_sql / "1.json",
    )
]

params_run = [
    (
        filepath_VTL / "2.vtl",
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        [filepath_csv / "DS_1.csv", filepath_csv / "DS_2.csv"],
        filepath_ValueDomains / "VD_1.json",
        filepath_sql / "1.json",
    ),
    (
        filepath_VTL / "2.vtl",
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        {"DS_1": filepath_csv / "custom_name_DS_1.csv", "DS_2": filepath_csv / "DS_2.csv"},
        filepath_ValueDomains / "VD_1.json",
        filepath_sql / "1.json",
    ),
]

params_schema = [(filepath_json / "DS_Schema.json")]

param_id_null = [((filepath_json / "DS_ID_null.json"), "Identifier Id_1 cannot be nullable")]

param_wrong_role = [((filepath_json / "DS_Role_wrong.json"), "0-2-1-1")]

param_wrong_data_type = [((filepath_json / "DS_wrong_datatype.json"), "0-2-1-1")]

param_viral_attr = [((filepath_json / "DS_Viral_attr.json"), "0-2-1-1")]

params_check_script = [
    (
        (
            TransformationScheme(
                id="TS1",
                version="1.0",
                agency="MD",
                vtl_version="2.1",
                items=[
                    Transformation(
                        id="T1",
                        uri=None,
                        urn=None,
                        name=None,
                        description=None,
                        expression="DS_1 + DS_2",
                        is_persistent=False,
                        result="DS_r",
                        annotations=(),
                    ),
                    Transformation(
                        id="T2",
                        uri=None,
                        urn=None,
                        name="simple",
                        description="addition",
                        expression="DS_1 + DS_3",
                        is_persistent=True,
                        result="DS_r2",
                        annotations=(),
                    ),
                    Transformation(
                        id="T3",
                        uri=None,
                        urn="sdmx:org.sdmx.infomodel.datastructure.DataStructure=BIS:BIS_DER(1.0)",
                        name=None,
                        description=None,
                        expression="left_join (securities_static, securities_dynamic using securityId);",
                        is_persistent=False,
                        result="DS_r3",
                        annotations=(),
                    ),
                ],
            ),
            (filepath_VTL / "check_script_reference.vtl"),
        )
    )
]

params_generate_sdmx = [
    ("DS_r := DS_1 + DS_2;", "MD", "1.0"),
    ("DS_r <- DS_1 + 1;", "MD", "1.0"),
    (
        """
    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
        FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
        sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
        sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1
        end datapoint ruleset;
        DS_r := check_datapoint (BOP, signValidation);
    """,
        "MD",
        "1.0",
    ),
    (
        """
        define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

        DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset);
        """,
        "MD",
        "1.0",
    ),
    (
        """
        define operator suma (ds1 dataset, ds2 dataset)
            returns dataset is
            ds1 + ds2
        end operator;
        DS_r := suma(ds1, ds2);
    """,
        "MD",
        "1.0",
    ),
]

params_run_with_scalars = [(filepath_json / "DS_3.json", filepath_csv / "DS_3.csv")]

params_validate_vd_sql_schema = [
    (
        filepath_json / "value_domain_schema.json",
        filepath_json / "external_routines_schema.json",
        filepath_ValueDomains / "VD_1.json",
        filepath_sql / "1.json",
    )
]

params_invalid_vd = [
    pytest.param(
        filepath_json / "value_domain_schema.json",
        filepath_ValueDomains / "VD_wrong_key.json",
        id="wrong_key",
    ),
    pytest.param(
        filepath_json / "value_domain_schema.json",
        filepath_ValueDomains / "VD_wrong_setlist.json",
        id="wrong_setlist",
    ),
    pytest.param(
        filepath_json / "value_domain_schema.json",
        filepath_ValueDomains / "VD_wrong_type.json",
        id="wrong_type",
    ),
    pytest.param(
        filepath_json / "value_domain_schema.json",
        filepath_ValueDomains / "VD_wrong_values.json",
        id="wrong_values",
    ),
]
params_invalid_sql = [
    pytest.param(
        filepath_json / "external_routines_schema.json",
        filepath_sql / "ext_routine_wrong_key.json",
        id="wrong_key",
    ),
    pytest.param(
        filepath_json / "external_routines_schema.json",
        filepath_sql / "ext_routine_wrong_query.json",
        id="wrong_query",
    ),
]


params_validate_ds = [
    (
        filepath_json / "DS_1.json",
        {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Id_2": ["A", "B"], "Me_1": [10, 20]})},
        True,
        None,
    ),
    (
        filepath_json / "DS_1.json",
        {"DS_1": pd.DataFrame({"wrong_col": [1, 2]})},
        False,
        # pandas reports the missing components first, DuckDB the extra column
        "0-3-1-5|0-3-1-15",
    ),
    (filepath_json / "DS_1.json", None, True, None),
    (
        filepath_json / "DS_1.json",
        {"DS_non_exist": pd.DataFrame({"Id_1": [1], "Me_1": [2]})},
        False,
        "Not found dataset DS_non_exist in datastructures.",
    ),
    (
        [filepath_json / "DS_1.json"],
        {"DS_1": pd.DataFrame({"Id_1": [1], "Id_2": ["A"], "Me_1": [10]})},
        True,
        None,
    ),
    (
        filepath_json / "DS_1.json",
        {"DS_1": pd.DataFrame({"Id_1": [1], "Id_2": ["A"], "Me_1": [10]})},
        True,
        None,
    ),
    (
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        [filepath_csv / "DS_1.csv", filepath_csv / "DS_2.csv"],
        True,
        None,
    ),
    (
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        {"DS_1": filepath_csv / "DS_1.csv", "DS_2": filepath_csv / "DS_2.csv"},
        True,
        None,
    ),
    (
        [filepath_json / "DS_1.json"],
        {"DS_1": pd.DataFrame({"wrong": [1]})},
        False,
        "0-3-1-5|0-3-1-15",
    ),
    (
        [filepath_json / "DS_1.json", filepath_json / "DS_2.json"],
        {"DS_1": filepath_csv / "custom_name_DS_1.csv", "DS_2": filepath_csv / "DS_2.csv"},
        True,
        None,
    ),
    (
        [filepath_json / "DS_1.json"],
        {"DS_1": filepath_csv / "DS_1_invalid.csv"},
        False,
        "On Dataset DS_1 loading: not possible to cast column Me_1 to Number.",
    ),
    # A DataFrame is read on the rules of the pandas loader on both engines, which
    # name the column that failed and its type (issue #1284): decimals in an Integer
    # are refused rather than rounded, a boolean is a Number (1.0 or 0.0), and a value
    # DuckDB cannot cast is traced back to its column
    (
        filepath_json / "GH_1284.json",
        {"DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [1.5, 2.5, -0.7], "Me_2": 1.0})},
        False,
        "not possible to cast column Me_1 to Integer",
    ),
    (
        filepath_json / "GH_1284.json",
        {"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [1], "Me_2": [True]})},
        True,
        None,
    ),
    (
        filepath_json / "GH_1284.json",
        {"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [1], "Me_2": ["0x10"]})},
        False,
        "not possible to cast column Me_2 to Number",
    ),
]

params_validate_vd = [
    (filepath_ValueDomains / "VD_1.json", True),
    (filepath_ValueDomains / "VD_wrong_key.json", False),
]

params_validate_sql = [
    (filepath_sql / "1.json", True),
    (filepath_sql / "ext_routine_wrong_key.json", False),
    (filepath_sql / "ext_routine_wrong_query.json", False),
]


@pytest.mark.parametrize("input", ext_params_OK)
def test_load_external_routine(input):
    result = load_external_routines(input)
    reference = {
        "1": ExternalRoutine(
            dataset_names=["BNFCRS_TRNSFRS", "BNFCRS_TRNSFRS_CMMN_INSTRMNTS_4"],
            query="SELECT date(DT_RFRNC) as DT_RFRNC, PRSPCTV_ID, "
            "INSTRMNT_UNQ_ID, BNFCRS_CNTRPRTY_ID, "
            "TRNSFR_CNTRPRTY_ID, BNFCR_ID, TRNSFR_ID FROM "
            "BNFCRS_TRNSFRS WHERE INSTRMNT_UNQ_ID NOT "
            "IN(SELECT INSTRMNT_UNQ_ID FROM "
            "BNFCRS_TRNSFRS_CMMN_INSTRMNTS_4);",
            name="1",
        )
    }

    assert result == reference


@pytest.mark.parametrize("input, code", ext_params_wrong)
def test_load_external_routine_with_wrong_params(input, code):
    with pytest.raises(Exception, match=code):
        load_external_routines(input)


@pytest.mark.parametrize("input, expression", input_vtl_params_OK)
def test_load_input_vtl(input, expression):
    text = load_vtl(input)
    result = text
    assert result == expression


@pytest.mark.parametrize("input, code", input_vtl_error_params)
def test_load_wrong_inputs_vtl(input, code):
    with pytest.raises(Exception, match=code):
        load_vtl(input)


@pytest.mark.parametrize("input", input_vd_OK)
def test_load_input_vd(input):
    result = load_value_domains(input)
    reference = ValueDomain(name="AnaCreditCountries", setlist=["AT", "BE", "CY"], type=String)
    assert "AnaCreditCountries" in result
    assert result["AnaCreditCountries"] == reference


@pytest.mark.parametrize("input, code", input_vd_error_params)
def test_load_wrong_inputs_vd(input, code):
    with pytest.raises(Exception, match=code):
        load_value_domains(input)


@pytest.mark.parametrize("datastructure", load_datasets_input_params_OK)
def test_load_datastructures(datastructure):
    datasets, scalars = load_datasets(datastructure)
    reference_dataset = Dataset(
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
    reference_scalar = Scalar(
        name="sc_1",
        data_type=DataTypes.Integer,
        value=None,
    )

    assert "DS_1" in datasets
    assert datasets["DS_1"] == reference_dataset

    assert "sc_1" in scalars
    assert scalars["sc_1"] == reference_scalar


@pytest.mark.parametrize("input, code", load_datasets_wrong_input_params)
def test_load_wrong_inputs_datastructures(input, code):
    with pytest.raises(Exception, match=code):
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
    "data_structure,expected_components",
    [
        # Test with 'type' key in DataStructure
        (
            {
                "datasets": [
                    {
                        "name": "test_ds",
                        "DataStructure": [
                            {
                                "name": "id1",
                                "type": "Integer",
                                "role": "Identifier",
                                "nullable": False,
                            },
                            {
                                "name": "measure1",
                                "type": "Number",
                                "role": "Measure",
                                "nullable": True,
                            },
                        ],
                    }
                ]
            },
            {"id1": DataTypes.Integer, "measure1": DataTypes.Number},
        ),
        # Test with 'data_type' key for backward compatibility
        (
            {
                "datasets": [
                    {
                        "name": "test_ds",
                        "DataStructure": [
                            {
                                "name": "id1",
                                "data_type": "String",
                                "role": "Identifier",
                                "nullable": False,
                            },
                            {
                                "name": "attr1",
                                "data_type": "Boolean",
                                "role": "Attribute",
                                "nullable": True,
                            },
                        ],
                    }
                ]
            },
            {"id1": DataTypes.String, "attr1": DataTypes.Boolean},
        ),
    ],
    ids=["type_key", "data_type_key_backward_compat"],
)
def test_load_datasets_datastructure_key_compatibility(data_structure, expected_components):
    """Test that load_datasets accepts both 'type' and 'data_type' keys in DataStructure"""
    datasets, scalars = load_datasets(data_structure)

    assert "test_ds" in datasets
    for comp_name, expected_type in expected_components.items():
        assert comp_name in datasets["test_ds"].components
        assert datasets["test_ds"].components[comp_name].data_type == expected_type


@pytest.mark.parametrize(
    "data_structure,expected_components",
    [
        # Test with 'type' key in structure components
        (
            {
                "structures": [
                    {
                        "name": "test_structure",
                        "components": [
                            {
                                "name": "id1",
                                "type": "Integer",
                                "role": "Identifier",
                                "nullable": False,
                            },
                            {
                                "name": "measure1",
                                "type": "String",
                                "role": "Measure",
                                "nullable": True,
                            },
                        ],
                    }
                ],
                "datasets": [{"name": "test_ds", "structure": "test_structure"}],
            },
            {"id1": DataTypes.Integer, "measure1": DataTypes.String},
        ),
        # Test with 'data_type' key for backward compatibility
        (
            {
                "structures": [
                    {
                        "name": "test_structure",
                        "components": [
                            {
                                "name": "id1",
                                "data_type": "Number",
                                "role": "Identifier",
                                "nullable": False,
                            },
                            {
                                "name": "attr1",
                                "data_type": "Boolean",
                                "role": "Attribute",
                                "nullable": True,
                            },
                        ],
                    }
                ],
                "datasets": [{"name": "test_ds", "structure": "test_structure"}],
            },
            {"id1": DataTypes.Number, "attr1": DataTypes.Boolean},
        ),
    ],
    ids=["type_key", "data_type_key_backward_compat"],
)
def test_load_datasets_structure_key_compatibility(data_structure, expected_components):
    """Test that load_datasets accepts both 'type' and 'data_type' keys in structure components"""
    datasets, scalars = load_datasets(data_structure)

    assert "test_ds" in datasets
    for comp_name, expected_type in expected_components.items():
        assert comp_name in datasets["test_ds"].components
        assert datasets["test_ds"].components[comp_name].data_type == expected_type


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
    }

    assert result == reference


@pytest.mark.parametrize(
    "script, data_structures, datapoints, value_domains, external_routines", params_run
)
def test_run(script, data_structures, datapoints, value_domains, external_routines):
    result = run(
        script,
        data_structures,
        datapoints,
        value_domains,
        external_routines,
        return_only_persistent=False,
        use_duckdb=_use_duckdb_backend(),
    )
    reference = {
        "DS_r": Dataset(
            name="DS_r",
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
            data=pd.DataFrame(
                columns=["Id_1", "Id_2", "Me_1"],
                index=[0, 1],
                data=[(1, "A", 2), (1, "B", 4)],
            ),
        ),
        "DS_r2": Dataset(
            name="DS_r2",
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
            data=pd.DataFrame(
                columns=["Id_1", "Id_2", "Me_1"],
                index=[0, 1],
                data=[(1, "A", 3), (1, "B", 6)],
            ),
        ),
    }

    assert result == reference


@pytest.mark.parametrize(
    "script, data_structures, datapoints, value_domains, external_routines", params_run
)
def test_run_only_persistent_results(
    script, data_structures, datapoints, value_domains, external_routines, tmp_path
):
    output_path = tmp_path

    result = run(
        script,
        data_structures,
        datapoints,
        value_domains,
        external_routines,
        output_folder=output_path,
        return_only_persistent=True,
        use_duckdb=_use_duckdb_backend(),
    )

    reference = {
        "DS_r2": Dataset(
            name="DS_r2",
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
    }

    assert result == reference
    files = list(output_path.iterdir())
    assert len(files) == 1
    assert set(result.keys()) == {"DS_r2"}
    expected_file = output_path / "DS_r2.csv"
    assert expected_file.exists()
    content = expected_file.read_text(encoding="utf-8").strip()
    assert content


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
        use_duckdb=_use_duckdb_backend(),
    )
    reference = {
        "DS_r2": Dataset(
            name="DS_r2",
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
            data=pd.DataFrame(
                columns=["Id_1", "Id_2", "Me_1"],
                index=[0, 1],
                data=[(1, "A", 3), (1, "B", 6)],
            ),
        )
    }

    assert result == reference


def test_readme_example():
    script = """
        DS_A := DS_1 * 10;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "Number",
                        "role": "Measure",
                        "nullable": True,
                    },
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10, 20, 30]})

    datapoints = {"DS_1": data_df}

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        return_only_persistent=False,
        use_duckdb=_use_duckdb_backend(),
    )

    assert run_result == {
        "DS_A": Dataset(
            name="DS_A",
            components={
                "Id_1": Component(
                    name="Id_1",
                    data_type=DataTypes.Integer,
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
            data=pd.DataFrame(
                columns=["Id_1", "Me_1"],
                index=[0, 1, 2],
                data=[(1, 100), (2, 200), (3, 300)],
            ),
        )
    }


def test_readme_run():
    script = """
        DS_A := DS_1 * 10;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "Number",
                        "role": "Measure",
                        "nullable": True,
                    },
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10, 20, 30]})

    datapoints = {"DS_1": data_df}

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        return_only_persistent=False,
        use_duckdb=_use_duckdb_backend(),
    )

    assert run_result == {
        "DS_A": Dataset(
            name="DS_A",
            components={
                "Id_1": Component(
                    name="Id_1",
                    data_type=DataTypes.Integer,
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
            data=pd.DataFrame(
                columns=["Id_1", "Me_1"],
                index=[0, 1, 2],
                data=[(1, 100), (2, 200), (3, 300)],
            ),
        )
    }


def test_readme_semantic_error():
    from vtlengine import semantic_analysis

    script = """
        DS_A := DS_1 * 10;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "String",
                        "role": "Measure",
                        "nullable": True,
                    },
                ],
            }
        ]
    }

    # Check error message
    with pytest.raises(SemanticError, match="1-1-1-2"):
        semantic_analysis(script=script, data_structures=data_structures)

    # Check output dataset on error message
    with pytest.raises(SemanticError, match="DS_A"):
        semantic_analysis(script=script, data_structures=data_structures)


def test_non_mandatory_fill_at():
    script = """
        DS_r := DS_1;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Id_2",
                        "type": "String",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "String",
                        "role": "Measure",
                        "nullable": True,
                    },
                    {
                        "name": "At_1",
                        "type": "String",
                        "role": "Attribute",
                        "nullable": True,
                    },
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 1, 2], "Id_2": ["A", "B", "A"], "Me_1": ["N", "N", "O"]})

    datapoints = {"DS_1": data_df}

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        return_only_persistent=False,
        use_duckdb=_use_duckdb_backend(),
    )

    assert run_result == {
        "DS_r": Dataset(
            name="DS_r",
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
                    data_type=DataTypes.String,
                    role=Role.MEASURE,
                    nullable=True,
                ),
                "At_1": Component(
                    name="At_1",
                    data_type=DataTypes.String,
                    role=Role.ATTRIBUTE,
                    nullable=True,
                ),
            },
            data=pd.DataFrame(
                columns=["Id_1", "Id_2", "Me_1", "At_1"],
                index=[0, 1, 2],
                data=pd.DataFrame(
                    {
                        "Id_1": [1, 1, 2],
                        "Id_2": ["A", "B", "A"],
                        "Me_1": ["N", "N", "O"],
                        "At_1": [None, None, None],
                    }
                ),
            ),
        )
    }


def test_non_mandatory_fill_me():
    script = """
        DS_r := DS_1;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Id_2",
                        "type": "String",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "String",
                        "role": "Measure",
                        "nullable": True,
                    },
                    {
                        "name": "At_1",
                        "type": "String",
                        "role": "Attribute",
                        "nullable": True,
                    },
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 1, 2], "Id_2": ["A", "B", "A"], "At_1": ["N", "N", "O"]})

    datapoints = {"DS_1": data_df}

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        return_only_persistent=False,
        use_duckdb=_use_duckdb_backend(),
    )

    assert run_result == {
        "DS_r": Dataset(
            name="DS_r",
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
                    data_type=DataTypes.String,
                    role=Role.MEASURE,
                    nullable=True,
                ),
                "At_1": Component(
                    name="At_1",
                    data_type=DataTypes.String,
                    role=Role.ATTRIBUTE,
                    nullable=True,
                ),
            },
            data=pd.DataFrame(
                columns=["Id_1", "Id_2", "Me_1", "At_1"],
                index=[0, 1, 2],
                data=pd.DataFrame(
                    {
                        "Id_1": [1, 1, 2],
                        "Id_2": ["A", "B", "A"],
                        "Me_1": [None, None, None],
                        "At_1": ["N", "N", "O"],
                    }
                ),
            ),
        )
    }


def test_mandatory_at_error():
    exception_code = "0-3-1-5"

    script = """
        DS_r := DS_1;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Id_2",
                        "type": "String",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "String",
                        "role": "Measure",
                        "nullable": True,
                    },
                    {
                        "name": "At_1",
                        "type": "String",
                        "role": "Attribute",
                        "nullable": False,
                    },
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 1, 2], "Id_2": ["A", "B", "A"], "Me_1": ["N", "N", "O"]})

    datapoints = {"DS_1": data_df}

    with pytest.raises(DataLoadError) as context:
        run(script=script, data_structures=data_structures, datapoints=datapoints)
    result = exception_code == str(context.value.args[1])
    if result is False:
        print(f"\n{exception_code} != {context.value.args[1]}")
    assert result


def test_mandatory_me_error():
    exception_code = "0-3-1-5"

    script = """
        DS_r := DS_1;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Id_2",
                        "type": "String",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "String",
                        "role": "Measure",
                        "nullable": False,
                    },
                    {
                        "name": "At_1",
                        "type": "String",
                        "role": "Attribute",
                        "nullable": True,
                    },
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 1, 2], "Id_2": ["A", "B", "A"], "At_1": ["N", "N", "O"]})

    datapoints = {"DS_1": data_df}

    with pytest.raises(DataLoadError) as context:
        run(script=script, data_structures=data_structures, datapoints=datapoints)
    result = exception_code == str(context.value.args[1])
    if result is False:
        print(f"\n{exception_code} != {context.value.args[1]}")
    assert result


@pytest.mark.parametrize("data_structure", params_schema)
def test_load_data_structure_with_new_schema(data_structure):
    datasets, _ = load_datasets(data_structure)
    reference = Dataset(
        name="DS_Schema",
        components={
            "Id_1": Component(
                name="Id_1",
                data_type=DataTypes.String,
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
                data_type=DataTypes.Integer,
                role=Role.MEASURE,
                nullable=True,
            ),
            "At_1": Component(
                name="At_1",
                data_type=DataTypes.String,
                role=Role.VIRAL_ATTRIBUTE,
                nullable=True,
            ),
        },
        data=None,
    )
    assert "DS_Schema" in datasets
    assert datasets["DS_Schema"] == reference


@pytest.mark.parametrize("ds_r, error_message", param_id_null)
def test_load_data_structure_with_null_id(ds_r, error_message):
    with pytest.raises(InputValidationException, match=error_message):
        load_datasets(ds_r)


@pytest.mark.parametrize("ds_r, error_code", param_wrong_role)
def test_load_data_structure_with_wrong_role(ds_r, error_code):
    with pytest.raises(InputValidationException, match=error_code):
        load_datasets(ds_r)


@pytest.mark.parametrize("ds_r, error_code", param_wrong_data_type)
def test_load_data_structure_with_wrong_data_type(ds_r, error_code):
    with pytest.raises(InputValidationException, match=error_code):
        load_datasets(ds_r)


def test_check_script_with_string_input():
    script = "DS_r := DS_1 + DS_2;"
    result = _check_script(script)
    assert result == script


def test_check_script_invalid_input_type():
    with pytest.raises(
        InputValidationException,
        match="0-1-1-1",
    ):
        _check_script(12345)


@pytest.mark.parametrize("transformation_scheme, result_script", params_check_script)
def test_check_script_with_transformation_scheme(transformation_scheme, result_script):
    result = _check_script(transformation_scheme)
    with open(result_script, "r") as file:
        reference = file.read()
    assert prettify(result) == prettify(reference)


@pytest.mark.parametrize(
    "time_period_format, expected",
    [
        ("vtl", "2020M1"),
        ("sdmx_reporting", "2020-M01"),
        ("sdmx_gregorian", "2020-01"),
        ("natural", "2020-01"),
    ],
)
def test_scalar_time_period_output_format_with_output_folder(
    time_period_format, expected, tmp_path
):
    """A Time_Period scalar written to a folder carries the requested representation.

    The API only formats the results it hands back when nothing is written to disk, so
    a scalar going to a file used to keep whatever representation it had.
    """
    script = 'Sc_r <- cast("2020-M01", time_period);'
    run_result = run(
        script=script,
        data_structures={"datasets": []},
        datapoints={},
        output_folder=tmp_path,
        time_period_output_format=time_period_format,
        use_duckdb=_use_duckdb_backend(),
    )

    sc_csv = tmp_path / "_scalars.csv"
    assert sc_csv.exists()
    with open(sc_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["name", "value"]
    assert rows[1] == ["Sc_r", expected]
    # What is returned and what is written agree.
    assert run_result["Sc_r"].value == expected


def test_output_folder_does_not_mutate_shared_time_period_data(tmp_path):
    """Writing an alias to the output folder must not reformat the DataFrame it
    shares with its operand (GH #1042).

    save_datapoints applied the external Time_Period representation in place on
    the alias' data, which is the same DataFrame DS_1 holds (2000M1 external vs
    2000-M01 internal), so the later join matched no keys: full_join doubled the
    common Data Points and inner_join dropped them.
    """
    script = """
        A <- DS_1;
        DS_r <- full_join(DS_1, DS_2);
    """
    run(
        script=script,
        data_structures=[
            filepath_json / "GH_1042_DS_1.json",
            filepath_json / "GH_1042_DS_2.json",
        ],
        datapoints={
            "DS_1": filepath_csv / "GH_1042_DS_1.csv",
            "DS_2": filepath_csv / "GH_1042_DS_2.csv",
        },
        output_folder=tmp_path,
        use_duckdb=_use_duckdb_backend(),
    )
    lines = (tmp_path / "DS_r.csv").read_text().strip().splitlines()
    assert lines[0] == "Id_1,Id_2,Me_1,Me_2"
    assert sorted(lines[1:]) == [
        "1,2000M1,1,10",
        "2,2000M2,2,20",
        "3,2000M3,3,",
        "9,2020M1,,90",
    ]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1304_output_folder_releases_data(use_duckdb, tmp_path):
    """With an output folder, the results are written and returned without data (GH #1304).

    https://github.com/Meaningful-Data/vtlengine/issues/1304

    The pandas engine wrote DS_r.csv, but returned DS_r with its data and kept the data of
    every temporary result until the run ended, so writing to a folder saved no memory.
    """
    script = "\n".join(f"DS_{k + 1} := DS_{k} + 1;" for k in range(1, 11)) + "\nDS_r <- DS_11 + 1;"
    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [1.0, 2.0, 3.0]})

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints={"DS_1": data_df},
        output_folder=tmp_path,
        use_duckdb=use_duckdb,
    )

    assert list(run_result) == ["DS_r"]
    assert run_result["DS_r"].data is None
    assert [path.name for path in tmp_path.iterdir()] == ["DS_r.csv"]
    lines = (tmp_path / "DS_r.csv").read_text().strip().splitlines()
    assert lines[0] == "Id_1,Me_1"
    assert sorted(lines[1:]) == ["1,12", "2,13", "3,14"]


@pytest.mark.parametrize("data_structures, datapoints", params_run_with_scalars)
def test_run_with_scalars(data_structures, datapoints, tmp_path):
    script = """
        DS_r <- DS_3[filter Me_1 = sc_1];
        DS_r2 <- DS_3[sub Id_1 = 20];
        Sc_r <- sc_1 + sc_2 + 3 + sc_3;
        Sc_r2 <- sc_1 - sc_2;
        Sc_r3 <- null;
    """
    scalars = {"sc_1": 20, "sc_2": 5, "sc_3": 3}
    output_folder = tmp_path
    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        scalar_values=scalars,
        output_folder=output_folder,
        return_only_persistent=True,
        use_duckdb=_use_duckdb_backend(),
    )
    reference = {
        "DS_r": Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1",
                    data_type=DataTypes.Integer,
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
        "DS_r2": Dataset(
            name="DS_r2",
            components={
                "Me_1": Component(
                    name="Me_1",
                    data_type=DataTypes.Number,
                    role=Role.MEASURE,
                    nullable=True,
                ),
            },
            data=None,
        ),
        "Sc_r": Scalar(name="Sc_r", data_type=Integer, value=31),
        "Sc_r2": Scalar(name="Sc_r2", data_type=Integer, value=15),
        "Sc_r3": Scalar(name="Sc_r3", data_type=Null, value=None),
    }
    assert run_result == reference
    ds_csv = output_folder / "DS_r.csv"
    sc_csv = output_folder / "_scalars.csv"
    assert ds_csv.exists()
    assert sc_csv.exists()
    df = pd.read_csv(ds_csv)
    assert list(df.columns) == ["Id_1", "Me_1"]
    assert df.loc[0, "Id_1"] == 2
    assert df.loc[0, "Me_1"] == 20
    with open(sc_csv, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) == 4
    assert rows[0] == ["name", "value"]
    assert rows[1] == ["Sc_r", "31"]
    assert rows[2] == ["Sc_r2", "15"]
    assert rows[3] == ["Sc_r3", ""]
    assert all(isinstance(v, (Dataset, Scalar)) for v in run_result.values())


@pytest.mark.parametrize("data_structures, datapoints", params_run_with_scalars)
def test_run_with_scalar_being_none(data_structures, datapoints, tmp_path):
    script = """
        DS_r <- DS_3[filter Me_1 = sc_1];
        DS_r2 <- DS_3[sub Id_1 = 20];
        Sc_r <- sc_1 + sc_2 + 3 + sc_3;
    """
    scalars = {"sc_1": 20, "sc_2": 5, "sc_3": None}
    output_folder = tmp_path
    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        scalar_values=scalars,
        output_folder=output_folder,
        return_only_persistent=True,
        use_duckdb=_use_duckdb_backend(),
    )
    reference = {
        "DS_r": Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1",
                    data_type=DataTypes.Integer,
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
        "DS_r2": Dataset(
            name="DS_r2",
            components={
                "Me_1": Component(
                    name="Me_1",
                    data_type=DataTypes.Number,
                    role=Role.MEASURE,
                    nullable=True,
                ),
            },
            data=None,
        ),
        "Sc_r": Scalar(name="Sc_r", data_type=Integer, value=None),
    }
    assert run_result == reference
    ds_csv = output_folder / "DS_r.csv"
    sc_csv = output_folder / "_scalars.csv"
    assert ds_csv.exists()
    assert sc_csv.exists()
    df = pd.read_csv(ds_csv)
    assert list(df.columns) == ["Id_1", "Me_1"]
    assert df.loc[0, "Id_1"] == 2
    assert df.loc[0, "Me_1"] == 20
    with open(sc_csv, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0] == ["name", "value"]
    assert rows[1] == ["Sc_r", ""]


def test_script_with_component_working_as_scalar_and_component():
    script = """
            Me_2 <- 10;
            DS_r <- DS_1[filter Me_1 = Me_2];
        """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ],
        "scalars": [
            {
                "name": "Sc_1",
                "type": "Number",
            }
        ],
    }

    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10, 20, 30]})
    datapoints = {"DS_1": data_df}
    with pytest.raises(SemanticError, match="1-1-6-11"):
        run(
            script=script,
            data_structures=data_structures,
            datapoints=datapoints,
            return_only_persistent=True,
            use_duckdb=_use_duckdb_backend(),
        )


wrong_types_params = [
    ("string", "String"),  # Check lower case
    ("Nuber", "Number"),  # Check missing letter
    ("intger", "Integer"),  # Check lowercase and missing letter
    ("TimePeriod", "Time_Period"),  # Check underscore
    ("bool", ""),  # Has no closest marker
    ("dates", "Date"),  # Check plural form
    ("TimeInterval", "Time"),  # Check TimeInterval to Time
]


@pytest.mark.parametrize("wrong_type, correct_type", wrong_types_params)
def test_wrong_type_in_scalar_definition(wrong_type, correct_type):
    script = """
        sc_r <- sc_1;
    """

    data_structures = {
        "scalars": [
            {
                "name": "sc_1",
                "type": wrong_type,
            }
        ]
    }

    with pytest.raises(InputValidationException, match="0-2-1-1") as e:
        run(
            script=script,
            data_structures=data_structures,
            datapoints={},
        )
    assert wrong_type in e.value.args[0]
    assert correct_type in e.value.args[0]


@pytest.mark.parametrize(
    "path_vd_schema, path_ext_routine_schema, path_vd, path_sql", params_validate_vd_sql_schema
)
def test_validate_json_schema_on_vd_and_external_routine(
    path_vd_schema, path_ext_routine_schema, path_vd, path_sql
):
    with open(path_vd, "r") as f:
        vd_data = json.load(f)
    with open(path_sql, "r") as f:
        ext_routine_data = json.load(f)
    with open(path_vd_schema, "r") as f:
        vd_schema = json.load(f)
    with open(path_ext_routine_schema, "r") as f:
        ext_routine_schema = json.load(f)
    _validate_json(vd_data, vd_schema, kind="Value Domain")
    _validate_json(ext_routine_data, ext_routine_schema, kind="External Routine")


@pytest.mark.parametrize("path_schema, path_vd", params_invalid_vd)
def test_attempt_to_validate_invalid_vd(path_schema, path_vd):
    with open(path_vd, "r") as f:
        vd_data = json.load(f)
    with open(path_schema, "r") as f:
        vd_schema = json.load(f)
    with pytest.raises(InputValidationException, match="0-2-1-1"):
        _validate_json(vd_data, vd_schema, kind="Value Domain")


@pytest.mark.parametrize("path_schema, path_sql", params_invalid_sql)
def test_attempt_to_validate_invalid_sql(path_schema, path_sql):
    with open(path_sql, "r") as f:
        ext_routine_data = json.load(f)
    with open(path_schema, "r") as f:
        ext_routine_schema = json.load(f)
    try:
        _validate_json(ext_routine_data, ext_routine_schema, kind="External Routine")
    except InputValidationException:
        with pytest.raises(InputValidationException, match="0-2-1-1"):
            _validate_json(ext_routine_data, ext_routine_schema, kind="External Routine")
        return
    query = ext_routine_data.get("query")
    name = ext_routine_data.get("name", "test_routine")

    with pytest.raises(Exception, match="Invalid SQL query in external routine"):
        ExternalRoutine.from_sql_query(name, query)


def test_with_multiple_vd_and_ext_routines():
    script = """
      DS_r <- DS_1 [ calc Me_2:= cast(Me_1, string) in Countries];
      DS_r2 <- DS_1 [ calc Me_2:= cast(Me_1, string) in Countries_EU_Sample];
      DS_r3 <- eval(SQL_3(DS_1) language "SQL" returns dataset {identifier<integer> Id_1, measure<number> Me_1});
      DS_r4 <- eval(SQL_4(DS_1) language "SQL" returns dataset {identifier<integer> Id_1, measure<number> Me_1});
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    data_df = pd.DataFrame(
        {"Id_1": [2012, 2012, 2012], "Id_2": ["AT", "DE", "FR"], "Me_1": [0, 4, 9]}
    )

    datapoints = {"DS_1": data_df}

    external_routines = [
        {
            "name": "SQL_3",
            "query": "SELECT Id_1, COUNT(*) AS Me_1 FROM DS_1 GROUP BY Id_1;",
        },
        filepath_sql / "SQL_4.json",
    ]

    value_domains = [
        {"name": "Countries_EU_Sample", "setlist": ["DE", "FR", "IT"], "type": "String"},
        filepath_ValueDomains / "VD_2.json",
    ]

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        value_domains=value_domains,
        external_routines=external_routines,
        use_duckdb=_use_duckdb_backend(),
    )

    reference = {
        "DS_r": Dataset(
            name="DS_r",
            components={
                "Id_1": Component("Id_1", DataTypes.Integer, Role.IDENTIFIER, False),
                "Id_2": Component("Id_2", DataTypes.String, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", DataTypes.Number, Role.MEASURE, True),
                "Me_2": Component("Me_2", DataTypes.Boolean, Role.MEASURE, True),
            },
            data=pd.DataFrame(
                {
                    "Id_1": [2012, 2012, 2012],
                    "Id_2": ["AT", "DE", "FR"],
                    "Me_1": [0.0, 4.0, 9.0],
                    "Me_2": [False, False, False],
                }
            ),
        ),
        "DS_r2": Dataset(
            name="DS_r2",
            components={
                "Id_1": Component("Id_1", DataTypes.Integer, Role.IDENTIFIER, False),
                "Id_2": Component("Id_2", DataTypes.String, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", DataTypes.Number, Role.MEASURE, True),
                "Me_2": Component("Me_2", DataTypes.Boolean, Role.MEASURE, True),
            },
            data=pd.DataFrame(
                {
                    "Id_1": [2012, 2012, 2012],
                    "Id_2": ["AT", "DE", "FR"],
                    "Me_1": [0.0, 4.0, 9.0],
                    "Me_2": [False, False, False],
                }
            ),
        ),
        "DS_r3": Dataset(
            name="DS_r3",
            components={
                "Id_1": Component("Id_1", DataTypes.Integer, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", DataTypes.Number, Role.MEASURE, True),
            },
            data=pd.DataFrame(
                {
                    "Id_1": [2012],
                    "Me_1": [3.0],
                }
            ),
        ),
        "DS_r4": Dataset(
            name="DS_r4",
            components={
                "Id_1": Component("Id_1", DataTypes.Integer, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", DataTypes.Number, Role.MEASURE, True),
            },
            data=pd.DataFrame(
                {
                    "Id_1": [2012, 2012, 2012],
                    "Me_1": [0.0, 4.0, 9.0],
                }
            ),
        ),
    }
    assert run_result["DS_r"] == reference["DS_r"]
    assert run_result["DS_r2"] == reference["DS_r2"]
    assert run_result["DS_r3"] == reference["DS_r3"]
    assert run_result["DS_r4"] == reference["DS_r4"]


def test_semantic_analysis_list_vd_ext_routines():
    external_routines = [
        {
            "name": "SQL_3",
            "query": "SELECT Id_1, COUNT(*) AS Me_1 FROM DS_1 GROUP BY Id_1;",
        },
        filepath_sql / "SQL_4.json",
    ]

    value_domains = [
        {"name": "Countries_EU_Sample", "setlist": ["DE", "FR", "IT"], "type": "String"},
        filepath_ValueDomains / "VD_2.json",
    ]
    script = """
          DS_r <- DS_1 [ calc Me_2:= cast(Me_1, string) in Countries];
          DS_r2 <- DS_1 [ calc Me_2:= cast(Me_1, string) in Countries_EU_Sample];
          DS_r3 <- eval(SQL_3(DS_1) language "SQL" returns dataset {identifier<integer> Id_1, measure<number> Me_1});
          DS_r4 <- eval(SQL_4(DS_1) language "SQL" returns dataset {identifier<integer> Id_1, measure<number> Me_1});
        """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    semantic_result = semantic_analysis(
        script=script,
        data_structures=data_structures,
        value_domains=value_domains,
        external_routines=external_routines,
    )

    reference = {
        "DS_r": Dataset(
            name="DS_r",
            components={
                "Id_1": Component("Id_1", DataTypes.Integer, Role.IDENTIFIER, False),
                "Id_2": Component("Id_2", DataTypes.String, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", DataTypes.Number, Role.MEASURE, True),
                "Me_2": Component("Me_2", DataTypes.Boolean, Role.MEASURE, True),
            },
            data=None,
        ),
        "DS_r2": Dataset(
            name="DS_r2",
            components={
                "Id_1": Component("Id_1", DataTypes.Integer, Role.IDENTIFIER, False),
                "Id_2": Component("Id_2", DataTypes.String, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", DataTypes.Number, Role.MEASURE, True),
                "Me_2": Component("Me_2", DataTypes.Boolean, Role.MEASURE, True),
            },
            data=None,
        ),
        "DS_r3": Dataset(
            name="DS_r3",
            components={
                "Id_1": Component("Id_1", DataTypes.Integer, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", DataTypes.Number, Role.MEASURE, True),
            },
            data=None,
        ),
        "DS_r4": Dataset(
            name="DS_r4",
            components={
                "Id_1": Component("Id_1", DataTypes.Integer, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", DataTypes.Number, Role.MEASURE, True),
            },
            data=None,
        ),
    }
    assert semantic_result["DS_r"] == reference["DS_r"]
    assert semantic_result["DS_r2"] == reference["DS_r2"]
    assert semantic_result["DS_r3"] == reference["DS_r3"]
    assert semantic_result["DS_r4"] == reference["DS_r4"]


@pytest.mark.parametrize(
    "value_domains_input",
    [
        [
            filepath_ValueDomains / "VD_1.json",
            filepath_ValueDomains / "VD_2.json",
        ]
    ],
)
def test_loading_list_multiple_value_domains(value_domains_input):
    value_domains_loaded = load_value_domains(value_domains_input)
    assert "Countries" in value_domains_loaded
    assert "AnaCreditCountries" in value_domains_loaded


@pytest.mark.parametrize("path_vd, is_valid", params_validate_vd)
def test_validate_vd(path_vd, is_valid):
    with open(path_vd, "r") as f:
        vd_data = json.load(f)
    if is_valid:
        try:
            validate_value_domain(vd_data)
        except Exception:
            with pytest.raises(Exception, match="The given json does not follow the schema."):
                validate_value_domain(vd_data)


@pytest.mark.parametrize("path_sql, is_valid", params_validate_sql)
def test_validate_sql(path_sql, is_valid):
    with open(path_sql, "r") as f:
        ext_routine_data = json.load(f)
    if is_valid:
        try:
            validate_external_routine(ext_routine_data)
        except Exception:
            with pytest.raises(Exception, match="The given json does not follow the schema."):
                validate_external_routine(ext_routine_data)


@pytest.mark.parametrize("use_duckdb", [False, True])
@pytest.mark.parametrize("ds_input, dp_input, is_valid, message", params_validate_ds)
def test_validate_dataset(ds_input, dp_input, is_valid, message, use_duckdb):
    if isinstance(ds_input, list):
        ds_data = []
        for item in ds_input:
            if isinstance(item, Path):
                with open(item, "r", encoding="utf-8") as f:
                    ds_data.append(json.load(f))
            else:
                ds_data.append(item)
    elif isinstance(ds_input, Path):
        with open(ds_input, "r", encoding="utf-8") as f:
            ds_data = json.load(f)
    else:
        ds_data = ds_input
    if is_valid:
        validate_dataset(ds_data, dp_input, use_duckdb=use_duckdb)
    else:
        with pytest.raises(Exception, match=message):
            validate_dataset(ds_data, dp_input, use_duckdb=use_duckdb)


params_load_errors_on_every_path = [
    ("GH_1216_1", "0-3-1-18", "On Dataset DS_1 loading: line 2 of the file does not split"),
    ("GH_1216_2", "0-3-1-18", "On Dataset DS_1 loading: line 3 of the file does not split"),
    ("GH_1216_3", "0-3-1-5", "On Dataset DS_1 loading: Component Me_1 is missing in Datapoints."),
    ("GH_1216_4", "0-3-1-5", "On Dataset DS_1 loading: Component Me_1 is missing in Datapoints."),
]


@pytest.mark.parametrize("code, error_code, message", params_load_errors_on_every_path)
def test_load_errors_on_every_path(code, error_code, message):
    """validate_dataset and run on both engines reject the same datapoints with the same
    error, which names the dataset and not the file (issue #1216)."""
    with open(filepath_json / f"{code}.json", "r", encoding="utf-8") as f:
        data_structures = json.load(f)
    datapoints = {"DS_1": filepath_csv / f"{code}.csv"}
    calls = [
        lambda: validate_dataset(data_structures, datapoints),
        lambda: validate_dataset(data_structures, datapoints, use_duckdb=True),
        lambda: run(script="DS_r := DS_1;", data_structures=data_structures, datapoints=datapoints),
        lambda: run(
            script="DS_r := DS_1;",
            data_structures=data_structures,
            datapoints=datapoints,
            use_duckdb=True,
        ),
    ]
    for call in calls:
        with pytest.raises(DataLoadError, match=message) as context:
            call()
        assert context.value.args[1] == error_code


def test_run_error_on_extra_dataframe_columns():
    """Extra columns in the input DataFrame that are not in the DataStructure raise an error."""
    script = "DS_r <- DS_1;"
    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    datapoints = {
        "DS_1": pd.DataFrame(
            {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "Extra_Col": ["a", "b", "c"]}
        )
    }

    with pytest.raises(DataLoadError, match="0-3-1-15"):
        run(script=script, data_structures=data_structures, datapoints=datapoints)


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
@pytest.mark.parametrize("input_kind", ["dataframe", "csv", "parquet"])
def test_run_error_on_extra_columns(tmp_path, input_kind, use_duckdb):
    """Whatever carries the input, a column outside the DataStructure is rejected.

    The DuckDb engine selected the DataStructure components and read past the rest,
    so it ran a script the pandas engine refused to load (issue #1053).
    """
    if input_kind == "parquet" and not use_duckdb:
        pytest.skip("the pandas loader reads a Parquet file as a CSV and fails earlier")

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    df = pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "Me_2": ["a", "b"]})
    if input_kind == "dataframe":
        datapoints = {"DS_1": df}
    elif input_kind == "csv":
        csv_path = tmp_path / "DS_1.csv"
        df.to_csv(csv_path, index=False)
        datapoints = {"DS_1": str(csv_path)}
    else:
        parquet_path = tmp_path / "DS_1.parquet"
        df.to_parquet(parquet_path, index=False)
        datapoints = {"DS_1": str(parquet_path)}

    with pytest.raises(DataLoadError, match="0-3-1-15"):
        run(
            script="DS_r <- DS_1 * 10;",
            data_structures=data_structures,
            datapoints=datapoints,
            use_duckdb=use_duckdb,
        )


_DS_1_STRUCTURE = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


def _datapoint_files(tmp_path):
    """A folder holding DS_1.csv, and a CSV whose name matches no dataset."""
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "DS_1.csv").write_text("Id_1,Me_1\n1,10\n")
    unknown = tmp_path / "UNKNOWN.csv"
    unknown.write_text("Id_1,Me_1\n1,10\n")
    return folder, unknown


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
def test_run_rejects_a_datapoint_path_that_does_not_exist(tmp_path, use_duckdb):
    """A path naming data that never arrived is reported, not read as no data.

    The DuckDb engine loaded an empty dataset, so the script ran to completion over
    nothing (issue #1061).
    """
    with pytest.raises(DataLoadError, match="0-3-1-1"):
        run(
            script="DS_r <- DS_1;",
            data_structures=_DS_1_STRUCTURE,
            datapoints={"DS_1": tmp_path / "does_not_exist.csv"},
            use_duckdb=use_duckdb,
        )


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
def test_run_reads_the_datapoint_files_of_a_directory(tmp_path, use_duckdb):
    """A directory stands for the datapoint files in it."""
    folder, _ = _datapoint_files(tmp_path)

    result = run(
        script="DS_r <- DS_1;",
        data_structures=_DS_1_STRUCTURE,
        datapoints=folder,
        use_duckdb=use_duckdb,
    )

    assert result["DS_r"].data.to_dict("records") == [{"Id_1": 1, "Me_1": 10.0}]


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
def test_run_rejects_a_datapoint_file_naming_no_dataset(tmp_path, use_duckdb):
    """A file whose name matches no dataset is reported, not dropped."""
    _, unknown = _datapoint_files(tmp_path)

    with pytest.raises(InputValidationException, match="Not found dataset UNKNOWN"):
        run(
            script="DS_r <- DS_1;",
            data_structures=_DS_1_STRUCTURE,
            datapoints=[unknown],
            use_duckdb=use_duckdb,
        )


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
@pytest.mark.parametrize("value", [None, 42], ids=["none", "number"])
def test_run_rejects_a_datapoint_dictionary_value_that_is_no_input(value, use_duckdb):
    """A dictionary value that is neither a Path nor a DataFrame names no data.

    The wording asked for all Paths or all DataFrames until issue #1072 dropped that
    restriction, so what is refused now is a value that names no Data Points at all.
    """
    with pytest.raises(InputValidationException, match="Must be DataFrame, Path, or string"):
        run(
            script="DS_r <- DS_1;",
            data_structures=_DS_1_STRUCTURE,
            datapoints={"DS_1": value},
            use_duckdb=use_duckdb,
        )


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
@pytest.mark.parametrize("marker", ["ACTION", "STRUCTURE", "STRUCTURE_ID", "DATAFLOW"])
def test_run_error_on_sdmx_marker_column_of_a_plain_csv(tmp_path, marker, use_duckdb):
    """A plain CSV carries no SDMX markers, so a column of that name is a column of it.

    The DuckDb engine took them for SDMX markers wherever they appeared, so a plain
    CSV holding an ACTION column lost the rows it marked deleted and the column with
    them, with no signal (issue #1064).
    """
    csv_path = tmp_path / "DS_1.csv"
    csv_path.write_text(f"Id_1,Me_1,{marker}\n1,10,I\n2,20,D\n")

    with pytest.raises(DataLoadError, match="0-3-1-15"):
        run(
            script="DS_r <- DS_1;",
            data_structures=_DS_1_STRUCTURE,
            datapoints={"DS_1": csv_path},
            use_duckdb=use_duckdb,
        )


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
def test_run_sdmx_csv_still_filters_the_rows_action_deletes(tmp_path, use_duckdb):
    """An SDMX-CSV file does carry them, and the rows ACTION deletes stay out."""
    csv_path = tmp_path / "DS_1.csv"
    csv_path.write_text(
        "STRUCTURE,STRUCTURE_ID,ACTION,Id_1,Me_1\n"
        "dataflow,MD:DS_1(1.0),I,1,10\n"
        "dataflow,MD:DS_1(1.0),D,2,20\n"
    )

    result = run(
        script="DS_r <- DS_1;",
        data_structures=_DS_1_STRUCTURE,
        datapoints={"DS_1": csv_path},
        use_duckdb=use_duckdb,
    )

    assert result["DS_r"].data.to_dict("records") == [{"Id_1": 1, "Me_1": 10.0}]


def test_run_error_on_missing_non_nullable_column():
    """Missing non-nullable columns in the input DataFrame raise an error."""
    script = "DS_r <- DS_1;"
    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": False},
                ],
            }
        ]
    }
    datapoints = {"DS_1": pd.DataFrame({"Id_1": [1, 2, 3]})}

    with pytest.raises(DataLoadError, match="0-3-1-5"):
        run(script=script, data_structures=data_structures, datapoints=datapoints)


def _chained_expression_script(operands: int) -> str:
    """A script whose last statement adds *operands* datasets together."""
    lines = [f"A_{i} := DS_1[calc Me_1 := Me_1 + {i}];" for i in range(operands)]
    lines.append("DS_r := " + " + ".join(f"A_{i}" for i in range(operands)) + ";")
    return "\n".join(lines)


_CHAIN_STRUCTURE = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


def test_recursion_headroom_restores_previous_limit():
    """The helper raises the limit only for its own scope."""
    previous = sys.getrecursionlimit()
    with recursion_headroom():
        assert sys.getrecursionlimit() >= MIN_RECURSION_LIMIT
    assert sys.getrecursionlimit() == previous


def test_recursion_headroom_keeps_a_higher_limit():
    """An already generous limit set by the caller is left untouched."""
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(MIN_RECURSION_LIMIT + 1000)
    try:
        with recursion_headroom():
            assert sys.getrecursionlimit() == MIN_RECURSION_LIMIT + 1000
        assert sys.getrecursionlimit() == MIN_RECURSION_LIMIT + 1000
    finally:
        sys.setrecursionlimit(previous)


def test_create_ast_handles_deeply_chained_expression():
    """A long ``a + b + c + ...`` chain builds without exhausting the stack.

    Each operand costs several frames while building the left-nested BinOp, so at
    CPython's default limit this raised RecursionError past roughly 300 operands.
    """
    chain = " + ".join(f"DS_{i}" for i in range(400))
    assert create_ast(f"DS_r := {chain};") is not None


def test_run_handles_deeply_chained_expression():
    """The same chain also transpiles and executes end to end."""
    script = _chained_expression_script(250)
    datapoints = {"DS_1": pd.DataFrame({"Id_1": ["a", "b"], "Me_1": [1.0, 2.0]})}

    result = run(
        script=script,
        data_structures=_CHAIN_STRUCTURE,
        datapoints=datapoints,
        return_only_persistent=False,
    )

    assert len(result["DS_r"].data) == 2


_TWO_DATASET_STRUCTURE = {
    "datasets": [
        {
            "name": name,
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
        for name in ("DS_1", "DS_2")
    ]
}


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
def test_run_reads_a_datapoint_dictionary_of_a_dataframe_and_a_file(tmp_path, use_duckdb):
    """A dictionary says where each dataset's Data Points are, one dataset at a time.

    Holding a DataFrame for one and a file for another was refused on the pandas
    engine, while the DuckDb engine read both (issue #1072).
    """
    csv_path = tmp_path / "DS_2.csv"
    csv_path.write_text("Id_1,Me_1\n1,5\n")

    result = run(
        script="DS_r <- DS_1 + DS_2;",
        data_structures=_TWO_DATASET_STRUCTURE,
        datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0]}), "DS_2": csv_path},
        use_duckdb=use_duckdb,
    )

    assert result["DS_r"].data.to_dict("records") == [{"Id_1": 1, "Me_1": 15.0}]


schema_validation_cases = [
    # Unknown top-level property
    (
        {"unknown_key": []},
        "Additional properties are not allowed",
    ),
    # Dataset missing 'name'
    (
        {
            "datasets": [
                {"DataStructure": [{"name": "Id_1", "type": "Integer", "role": "Identifier"}]}
            ]
        },
        "'name' is a required property",
    ),
    # Dataset has neither DataStructure nor structure
    (
        {"datasets": [{"name": "DS_1"}]},
        "is not valid under any of the given schemas",
    ),
    # Dataset has both DataStructure and structure (oneOf violation)
    (
        {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [{"name": "Id_1", "type": "Integer", "role": "Identifier"}],
                    "structure": "Struct_1",
                }
            ],
            "structures": [
                {
                    "name": "Struct_1",
                    "components": [{"name": "Id_1", "data_type": "Integer", "role": "Identifier"}],
                }
            ],
        },
        "is valid under each of",
    ),
    # Component missing 'role'
    (
        {"datasets": [{"name": "DS_1", "DataStructure": [{"name": "Id_1", "type": "Integer"}]}]},
        "'role' is a required property",
    ),
    # Component missing both 'type' and 'data_type'
    (
        {"datasets": [{"name": "DS_1", "DataStructure": [{"name": "Id_1", "role": "Identifier"}]}]},
        "is not valid under any of the given schemas",
    ),
    # Component with both 'type' and 'data_type'
    (
        {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {
                            "name": "Id_1",
                            "type": "Integer",
                            "data_type": "Integer",
                            "role": "Identifier",
                        }
                    ],
                }
            ]
        },
        "is valid under each of",
    ),
    # Component with invalid role enum
    (
        {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [{"name": "Id_1", "type": "Integer", "role": "BadRole"}],
                }
            ]
        },
        "'BadRole' is not one of",
    ),
    # Component with invalid data type enum
    (
        {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [{"name": "Id_1", "type": "BadType", "role": "Identifier"}],
                }
            ]
        },
        "'BadType' is not one of",
    ),
    # Component with nullable as non-boolean
    (
        {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {
                            "name": "Id_1",
                            "type": "Integer",
                            "role": "Identifier",
                            "nullable": "yes",
                        }
                    ],
                }
            ]
        },
        "is not of type 'boolean'",
    ),
    # Dataset name does not match the vtl-id pattern (bare name starting with '_')
    (
        {
            "datasets": [
                {
                    "name": "_bad",
                    "DataStructure": [{"name": "Id_1", "type": "Integer", "role": "Identifier"}],
                }
            ]
        },
        "does not match",
    ),
    # Scalar missing both 'type' and 'data_type'
    (
        {"scalars": [{"name": "sc_1"}]},
        "is not valid under any of the given schemas",
    ),
    # Scalar with invalid type enum
    (
        {"scalars": [{"name": "sc_1", "type": "BadType"}]},
        "'BadType' is not one of",
    ),
    # Structure missing 'components'
    (
        {
            "datasets": [{"name": "DS_1", "structure": "Struct_1"}],
            "structures": [{"name": "Struct_1"}],
        },
        "'components' is a required property",
    ),
]


@pytest.mark.parametrize("data_structures, expected_fragment", schema_validation_cases)
def test_data_structure_schema_validation(data_structures, expected_fragment):
    """Schema validation rejects malformed data_structures with informative messages."""
    with pytest.raises(InputValidationException, match="0-2-1-1") as excinfo:
        load_datasets(data_structures)
    assert expected_fragment in excinfo.value.args[0]


schema_failing_element_cases = [
    # Dataset error -> message includes "Dataset '<name>'"
    (
        {
            "datasets": [
                {
                    "name": "DS_named",
                    "DataStructure": [{"name": "Id_1", "type": "Integer"}],
                }
            ]
        },
        "Dataset 'DS_named'",
    ),
    # Scalar error -> message includes "Scalar '<name>'"
    (
        {"scalars": [{"name": "sc_named", "type": "BadType"}]},
        "Scalar 'sc_named'",
    ),
    # Structure error -> message includes "Structure '<name>'"
    (
        {
            "datasets": [{"name": "DS_1", "structure": "Struct_named"}],
            "structures": [
                {
                    "name": "Struct_named",
                    "components": [{"name": "Id_1", "role": "BadRole", "data_type": "Integer"}],
                }
            ],
        },
        "Structure 'Struct_named'",
    ),
]


@pytest.mark.parametrize("data_structures, expected_fragment", schema_failing_element_cases)
def test_data_structure_schema_error_identifies_element(data_structures, expected_fragment):
    """The validation error message points to the specific dataset/scalar/structure that failed."""
    with pytest.raises(InputValidationException, match="0-2-1-1") as excinfo:
        load_datasets(data_structures)
    assert expected_fragment in excinfo.value.args[0]


def test_referenced_structure_not_found():
    """A dataset that references a non-existent structure raises a clear error."""
    data_structures = {
        "datasets": [{"name": "DS_1", "structure": "Missing"}],
        "structures": [
            {
                "name": "Other",
                "components": [{"name": "Id_1", "data_type": "Integer", "role": "Identifier"}],
            }
        ],
    }
    with pytest.raises(InputValidationException, match="0-2-1-1") as excinfo:
        load_datasets(data_structures)
    assert "Missing" in excinfo.value.args[0]


def test_GH_1326_load_value_domain_objects():
    """``ValueDomain`` objects are accepted, alone or mixed in a list (GH #1326).

    https://github.com/Meaningful-Data/vtlengine/issues/1326
    """
    value_domain = ValueDomain(name="VD_1", type=String, setlist=["DE", "FR"])
    assert load_value_domains(value_domain) == {"VD_1": value_domain}

    loaded = load_value_domains(
        [
            value_domain,
            {"name": "VD_2", "type": "Integer", "setlist": [1, 2]},
            filepath_ValueDomains / "VD_2.json",
        ]
    )
    assert loaded["VD_1"] == value_domain
    assert loaded["VD_2"] == ValueDomain(name="VD_2", type=Integer, setlist=[1, 2])
    assert "Countries" in loaded
    validate_value_domain([value_domain])


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize("as_list", [False, True])
def test_GH_1326_run_with_value_domain_object(use_duckdb, as_list):
    """``run`` and ``semantic_analysis`` take ``ValueDomain`` objects (GH #1326).

    https://github.com/Meaningful-Data/vtlengine/issues/1326
    """
    value_domain = ValueDomain(name="vd", type=String, setlist=["A", "B"])
    value_domains = [value_domain] if as_list else value_domain
    script = "DS_r <- DS_1 [filter Id_1 in vd];"
    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "String", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    datapoints = {"DS_1": pd.DataFrame({"Id_1": ["A", "C"], "Me_1": [1.0, 2.0]})}

    semantic_result = semantic_analysis(
        script=script, data_structures=data_structures, value_domains=value_domains
    )
    assert "DS_r" in semantic_result

    result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        value_domains=value_domains,
        use_duckdb=use_duckdb,
    )
    data = result["DS_r"].data
    assert data["Id_1"].tolist() == ["A"]
    assert data["Me_1"].tolist() == [1.0]


@pytest.mark.parametrize(
    "value_domain, value, type_",
    [
        (
            {"name": "VD_1", "type": "Date", "setlist": ["2020-01-01", "2020-13-45"]},
            "2020-13-45",
            "Date",
        ),
        (
            {"name": "VD_1", "type": "Time_Period", "setlist": ["2020Q1", "2020Q9"]},
            "2020Q9",
            "Time_Period",
        ),
        ({"name": "VD_1", "type": "Duration", "setlist": ["A", "X"]}, "X", "Duration"),
        ({"name": "VD_1", "type": "Time", "setlist": ["2020-01-01/2020-12-31", "X"]}, "X", "Time"),
    ],
)
def test_GH_1343_invalid_setlist_values(value_domain, value, type_):
    """Each value of a setlist must be a valid value of the type of the value domain (GH #1343).

    https://github.com/Meaningful-Data/vtlengine/issues/1343
    """
    message = f"Value {value} of the setlist of Value Domain VD_1 is not a valid {type_}"
    with pytest.raises(InputValidationException, match=message) as error:
        validate_value_domain(value_domain)
    assert error.value.args[1] == "0-1-1-37"


@pytest.mark.parametrize(
    "type_, setlist",
    [
        ("Time_Period", ["2020Q1", "2020-Q1"]),
        ("Date", ["2020-01-01", "2020-1-1"]),
    ],
)
def test_GH_1343_setlist_values_in_another_form(type_, setlist):
    """A setlist cannot repeat a value written in another form of its type (GH #1343)."""
    with pytest.raises(InputValidationException, match="0-1-1-21") as error:
        validate_value_domain({"name": "VD_1", "type": type_, "setlist": setlist})
    assert str(setlist) in error.value.args[0]


def test_GH_1343_valid_time_setlists():
    """Valid values of the time types are accepted (GH #1343)."""
    validate_value_domain(
        [
            {"name": "VD_1", "type": "Date", "setlist": ["2020-01-01", "2020-12-31"]},
            {"name": "VD_2", "type": "Time_Period", "setlist": ["2020Q1", "2020-M01", "2020"]},
            {"name": "VD_3", "type": "Duration", "setlist": ["A", "Q", "D"]},
            {"name": "VD_4", "type": "Time", "setlist": ["2020-01-01/2020-12-31"]},
        ]
    )


def test_GH_1343_repeated_names(tmp_path):
    """Two value domains cannot have the same name, given in a list or in a directory
    (GH #1343)."""
    first = {"name": "VD_1", "type": "String", "setlist": ["ES", "FR"]}
    second = {"name": "VD_1", "type": "String", "setlist": ["IT"]}
    with pytest.raises(InputValidationException, match="Value Domain VD_1 is defined more"):
        validate_value_domain([first, second])
    for name, value_domain in (("first.json", first), ("second.json", second)):
        (tmp_path / name).write_text(json.dumps(value_domain))
    with pytest.raises(InputValidationException, match="0-1-1-38"):
        load_value_domains(tmp_path)


@pytest.mark.parametrize(
    "name",
    ["string", "integer", "number", "boolean", "date", "time", "time_period", "duration", "scalar"],
)
def test_GH_1343_basic_type_names(name):
    """A value domain cannot be named as a basic scalar type, which a script cannot refer to
    as a value domain (GH #1343)."""
    with pytest.raises(InputValidationException, match="0-1-1-39"):
        validate_value_domain({"name": name, "type": "String", "setlist": ["ES"]})
    with pytest.raises(InputValidationException, match="0-1-1-39"):
        ValueDomain(name=name, type=String, setlist=["ES"])


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1343_run_rejects_an_unused_invalid_value_domain(use_duckdb):
    """The value domains that run and semantic_analysis load are checked even when the script
    does not use them, and a value domain named as a basic type with capitals can be used
    (GH #1343)."""
    structures = {"datasets": [], "scalars": [{"name": "SC_1", "type": "Date"}]}
    invalid = [{"name": "VD_1", "type": "Date", "setlist": ["2020-01-01", "2020-13-45"]}]
    with pytest.raises(InputValidationException, match="0-1-1-37"):
        semantic_analysis(script="SC_r <- SC_1;", data_structures=structures, value_domains=invalid)
    with pytest.raises(InputValidationException, match="0-1-1-37"):
        run(
            script="SC_r <- SC_1;",
            data_structures=structures,
            datapoints={},
            value_domains=invalid,
            scalar_values={"SC_1": "2020-01-01"},
            use_duckdb=use_duckdb,
        )
    result = run(
        script='SC_r <- "ES" in String;',
        data_structures={"datasets": []},
        datapoints={},
        value_domains=[{"name": "String", "type": "String", "setlist": ["ES"]}],
        use_duckdb=use_duckdb,
    )
    assert result["SC_r"].value is True


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_16(use_duckdb):
    """The legacy subset key of a Component or a scalar was ignored, so OTHER loaded into a
    Component declared String with subset VD_1. It is rejected with an input error that tells
    to name the value domain in the type, by run, semantic_analysis and validate_dataset."""
    value_domains = [{"name": "VD_1", "type": "String", "setlist": ["A", "B"]}]
    identifier = {"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False}
    measure = {"name": "Me_3", "role": "Measure", "type": "String", "subset": "VD_1"}
    dataset = {"datasets": [{"name": "DS_1", "DataStructure": [identifier, measure]}]}
    scalar = {"datasets": [], "scalars": [{"name": "SC_1", "type": "String", "subset": "VD_1"}]}
    datapoints = {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_3": ["A", "OTHER"]})}
    calls = [
        lambda: run("DS_r <- DS_1;", dataset, datapoints, value_domains, use_duckdb=use_duckdb),
        lambda: run("SC_r <- SC_1;", scalar, {}, value_domains, use_duckdb=use_duckdb),
        lambda: semantic_analysis("DS_r <- DS_1;", dataset, value_domains),
        lambda: semantic_analysis("SC_r <- SC_1;", scalar, value_domains),
        lambda: validate_dataset(
            dataset, datapoints, use_duckdb=use_duckdb, value_domains=value_domains
        ),
    ]
    for call in calls:
        with pytest.raises(InputValidationException) as error:
            call()
        assert error.value.args[1] == "0-2-1-1"
        assert "is not accepted; name the value domain in its type instead" in error.value.args[0]
        assert '("type": "VD_1")' in error.value.args[0]


@pytest.mark.parametrize(
    "type_name, value",
    [
        ("String", "A"),
        ("Number", 1.5),
        ("Integer", 1),
        ("Boolean", True),
        ("Date", "2024-01-05"),
        ("Time_Period", "2024Q1"),
        ("Time", "2024-01-01/2024-12-31"),
        ("Duration", "A"),
    ],
)
@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_18(use_duckdb, type_name, value):
    """ValueDomain.to_dict and to_json write the type name that the loaders read, for every
    basic type: a Time_Period value domain was written TimePeriod and a Time one TimeInterval,
    which ValueDomain.from_dict rejected with 0-1-1-24 and the loaders with 0-2-1-1."""
    value_domain = ValueDomain.from_dict({"name": "VD_1", "type": type_name, "setlist": [value]})
    written = value_domain.to_dict()
    assert written["type"] == type_name
    assert ValueDomain.from_dict(written) == value_domain
    assert ValueDomain.from_json(value_domain.to_json()) == value_domain
    validate_value_domain(written)
    structures = {"datasets": [], "scalars": [{"name": "SC_1", "type": "VD_1"}]}
    analysis = semantic_analysis("SC_r <- SC_1;", structures, written)
    assert analysis["SC_r"].to_dict()["type"] == "VD_1"
    result = run(
        "SC_r <- SC_1;",
        structures,
        {},
        written,
        scalar_values={"SC_1": value},
        use_duckdb=use_duckdb,
    )
    assert result["SC_r"].to_dict()["type"] == "VD_1"


_STRUCTURES_1349 = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Integer", "role": "Measure", "nullable": True},
            ],
        }
    ],
    "scalars": [{"name": "SC_4", "type": "Integer"}],
}

_ROUTINE_1349 = (
    'DS_r <- eval(SQL_1(DS_1) language "SQL" returns dataset'
    " { identifier<integer> Id_1, measure<integer> Me_1 });"
)

_SQL_1 = {"name": "SQL_1", "query": "SELECT Id_1, Me_1 FROM DS_1"}


def _datapoints_1349():
    return {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [1, 3]})}


def _run_1349(script, use_duckdb, **arguments):
    arguments = {"datapoints": _datapoints_1349(), **arguments}
    return run(script, _STRUCTURES_1349, use_duckdb=use_duckdb, **arguments)


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_19(use_duckdb, tmp_path):
    """value_domains given as a Path to one JSON file holding a list of value domains, or to a
    folder holding such a file, loads every value domain of the list, as the documentation
    says; it raised 0-1-1-23."""
    value_domains = [
        {"name": "VD_B", "type": "Boolean", "setlist": [True, False]},
        {"name": "VD_TP", "type": "Time_Period", "setlist": ["2024Q1", "2024Q2"]},
    ]
    folder = tmp_path / "value_domains"
    folder.mkdir()
    path = folder / "value_domains.json"
    path.write_text(json.dumps(value_domains))
    identifier = {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False}
    id_2 = {"name": "Id_2", "type": "VD_B", "role": "Identifier", "nullable": False}
    structures = {"datasets": [{"name": "DS_1", "DataStructure": [identifier, id_2]}]}
    datapoints = {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Id_2": [True, False]})}
    for given in (path, folder):
        validate_value_domain(given)
        assert sorted(load_value_domains(given)) == ["VD_B", "VD_TP"]
        result = run("DS_r <- DS_1;", structures, datapoints, given, use_duckdb=use_duckdb)
        ds_r = result["DS_r"]
        assert ds_r.components["Id_2"].to_dict()["type"] == "VD_B"
        assert ds_r.data.sort_values("Id_1")["Id_2"].tolist() == [True, False]
        analysis = semantic_analysis("DS_r <- DS_1;", structures, given)
        assert analysis["DS_r"].components["Id_2"].to_dict()["type"] == "VD_B"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_25(use_duckdb):
    """Datapoints given as a list holding a DataFrame is not an accepted form: both engines
    raise 0-1-1-2; DuckDB raised a raw AttributeError."""
    with pytest.raises(InputValidationException) as error:
        _run_1349("DS_r <- DS_1;", use_duckdb, datapoints=[_datapoints_1349()["DS_1"]])
    assert error.value.args[1] == "0-1-1-2"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_26(use_duckdb, tmp_path):
    """An output_folder that does not exist yet, nested in another missing folder, is created
    with its parents and the results are written there; pandas raised FileNotFoundError."""
    output_folder = tmp_path / "missing" / "deeper"
    _run_1349("DS_r <- DS_1;", use_duckdb, output_folder=output_folder)
    written = pd.read_csv(output_folder / "DS_r.csv").sort_values("Id_1")
    assert written["Me_1"].tolist() == [1, 3]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_27(use_duckdb, tmp_path):
    """An output_folder that is an existing file raises 0-3-1-2 on both engines; pandas raised
    a raw OSError and DuckDB a raw FileExistsError."""
    output_folder = tmp_path / "out.txt"
    output_folder.write_text("x")
    with pytest.raises(DataLoadError) as error:
        _run_1349("DS_r <- DS_1;", use_duckdb, output_folder=output_folder)
    assert error.value.args[1] == "0-3-1-2"
    assert output_folder.read_text() == "x"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_28(use_duckdb, tmp_path):
    """A missing output_folder with a file suffix (result.csv) raises 0-3-1-2 on both engines;
    DuckDB created a folder named result.csv and wrote the results into it."""
    output_folder = tmp_path / "result.csv"
    with pytest.raises(DataLoadError) as error:
        _run_1349("DS_r <- DS_1;", use_duckdb, output_folder=output_folder)
    assert error.value.args[1] == "0-3-1-2"
    assert not output_folder.exists()


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_28b(use_duckdb, tmp_path):
    """The datapoints are checked before the output_folder on both engines: a missing
    datapoint file raises 0-3-1-1 and creates no output folder, also when the output folder is
    bad. DuckDB checked the output folder first, so it created the missing folder before
    raising 0-3-1-1 and raised 0-3-1-2 for an existing file."""
    datapoints = {"DS_1": tmp_path / "missing.csv"}
    output_folder = tmp_path / "missing" / "deeper"
    with pytest.raises(DataLoadError) as error:
        _run_1349("DS_r <- DS_1;", use_duckdb, datapoints=datapoints, output_folder=output_folder)
    assert error.value.args[1] == "0-3-1-1"
    assert not (tmp_path / "missing").exists()
    output_file = tmp_path / "out.txt"
    output_file.write_text("x")
    with pytest.raises(DataLoadError) as error:
        _run_1349("DS_r <- DS_1;", use_duckdb, datapoints=datapoints, output_folder=output_file)
    assert error.value.args[1] == "0-3-1-1"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_28c(use_duckdb, tmp_path):
    """Datapoints whose values do not fit the structure (a text in an Integer Measure), given
    as a DataFrame or as a CSV file, raise 0-3-1-6 on both engines and create no output folder:
    the folder is created with the first result written. DuckDB created the folder before
    loading the DataFrame, pandas before loading the CSV file."""
    csv_path = tmp_path / "DS_1.csv"
    csv_path.write_text("Id_1,Me_1\n1,x\n")
    frame = pd.DataFrame({"Id_1": [1], "Me_1": ["x"]})
    output_folder = tmp_path / "missing" / "deeper"
    for datapoints in ({"DS_1": frame}, {"DS_1": csv_path}):
        with pytest.raises(DataLoadError) as error:
            _run_1349(
                "DS_r <- DS_1;", use_duckdb, datapoints=datapoints, output_folder=output_folder
            )
        assert error.value.args[1] == "0-3-1-6"
        assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_51(use_duckdb, tmp_path, monkeypatch):
    """An empty string as output_folder is no output folder, as None is: both engines write
    nothing and return the results with their data. pandas wrote the results into the current
    folder and returned them without data."""
    monkeypatch.chdir(tmp_path)
    result = _run_1349("DS_r <- DS_1;", use_duckdb, output_folder="")
    assert result["DS_r"].data.sort_values("Id_1")["Me_1"].tolist() == [1, 3]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_29(use_duckdb):
    """A list given as a value in scalar_values is not a scalar value: it raises 0-1-2-7 as a
    value of the wrong type does (case 30); it raised a raw ValueError."""
    with pytest.raises(InputValidationException) as error:
        _run_1349("DS_r <- DS_1 * SC_4;", use_duckdb, scalar_values={"SC_4": [1, 2]})
    assert error.value.args[1] == "0-1-2-7"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_30(use_duckdb):
    """Control of case 29: a text given as the value of an Integer scalar raises 0-1-2-7."""
    with pytest.raises(InputValidationException) as error:
        _run_1349("DS_r <- DS_1 * SC_4;", use_duckdb, scalar_values={"SC_4": "x"})
    assert error.value.args[1] == "0-1-2-7"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_31(use_duckdb, tmp_path):
    """external_routines given as a Path to a JSON file holding an array of routines loads
    every routine of the array, as the documentation says; it raised a raw TypeError."""
    path = tmp_path / "routines.json"
    path.write_text(json.dumps([_SQL_1, {**_SQL_1, "name": "SQL_2"}]))
    script = _ROUTINE_1349 + _ROUTINE_1349.replace("DS_r", "DS_s").replace("SQL_1", "SQL_2")
    result = _run_1349(script, use_duckdb, external_routines=path)
    for name in ("DS_r", "DS_s"):
        assert result[name].data.sort_values("Id_1")["Me_1"].tolist() == [1, 3]
    analysis = semantic_analysis(script, _STRUCTURES_1349, external_routines=path)
    assert sorted(analysis["DS_s"].components) == ["Id_1", "Me_1"]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_32(use_duckdb, tmp_path):
    """A value_domains .json file that is not valid JSON raises the input error 0-1-1-40
    naming the file; run and semantic_analysis raised a raw JSONDecodeError."""
    path = tmp_path / "broken.json"
    path.write_text("not json")
    calls = [
        lambda: _run_1349("DS_r <- DS_1;", use_duckdb, value_domains=path),
        lambda: semantic_analysis("DS_r <- DS_1;", _STRUCTURES_1349, path),
        lambda: validate_value_domain(path),
    ]
    for call in calls:
        with pytest.raises(InputValidationException) as error:
            call()
        assert error.value.args[1] == "0-1-1-40"
        assert "broken.json" in error.value.args[0]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_32b(use_duckdb, tmp_path):
    """The data structures read from a .json file that is not valid JSON raise the same input
    error 0-1-1-40 as the value domains of case 32; they raised a raw JSONDecodeError."""
    path = tmp_path / "broken.json"
    path.write_text("not json")
    calls = [
        lambda: run("DS_r <- DS_1;", path, _datapoints_1349(), use_duckdb=use_duckdb),
        lambda: semantic_analysis("DS_r <- DS_1;", path),
    ]
    for call in calls:
        with pytest.raises(InputValidationException) as error:
            call()
        assert error.value.args[1] == "0-1-1-40"
        assert "broken.json" in error.value.args[0]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_33(use_duckdb, tmp_path):
    """An external_routines .json file that is not valid JSON raises the input error 0-1-1-40
    naming the file; run and semantic_analysis raised a raw JSONDecodeError."""
    path = tmp_path / "broken.json"
    path.write_text("not json")
    calls = [
        lambda: _run_1349(_ROUTINE_1349, use_duckdb, external_routines=path),
        lambda: semantic_analysis(_ROUTINE_1349, _STRUCTURES_1349, external_routines=path),
        lambda: validate_external_routine(path),
    ]
    for call in calls:
        with pytest.raises(InputValidationException) as error:
            call()
        assert error.value.args[1] == "0-1-1-40"
        assert "broken.json" in error.value.args[0]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_34(use_duckdb, tmp_path):
    """A routine read from a file is registered under the name its JSON gives, not under the
    name of the file: routine.json holding SQL_1 serves eval(SQL_1(...)); it raised 1-3-5."""
    path = tmp_path / "routine.json"
    path.write_text(json.dumps(_SQL_1))
    result = _run_1349(_ROUTINE_1349, use_duckdb, external_routines=path)
    assert result["DS_r"].data.sort_values("Id_1")["Me_1"].tolist() == [1, 3]
    analysis = semantic_analysis(_ROUTINE_1349, _STRUCTURES_1349, external_routines=path)
    assert sorted(analysis["DS_r"].components) == ["Id_1", "Me_1"]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_52(use_duckdb, tmp_path):
    """Two external routines of one name raise 0-1-1-41, as two value domains of one name
    raise 0-1-1-38, whether they are given in a list, in one JSON file or in two files of a
    folder; the routine read last was used."""
    other = {**_SQL_1, "query": "SELECT Id_1, Me_1 * 10 AS Me_1 FROM DS_1"}
    array = tmp_path / "routines.json"
    array.write_text(json.dumps([_SQL_1, other]))
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps(_SQL_1))
    (folder / "b.json").write_text(json.dumps(other))
    for routines in ([_SQL_1, other], array, folder):
        calls = [
            lambda: _run_1349(_ROUTINE_1349, use_duckdb, external_routines=routines),
            lambda: semantic_analysis(_ROUTINE_1349, _STRUCTURES_1349, external_routines=routines),
            lambda: validate_external_routine(routines),
        ]
        for call in calls:
            with pytest.raises(InputValidationException) as error:
                call()
            assert error.value.args[1] == "0-1-1-41"
            assert "SQL_1" in error.value.args[0]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1349_57(use_duckdb, tmp_path):
    """The structure semantic_analysis writes for a pivot result, whose Measures are named
    2024-01-05 or A B, is read back as an input structure by run and validate_dataset, and a
    script reads those Measures by their quoted names; written between single quotes in the
    structure, a name names the same Component. The structure schema rejected those names."""

    def component(name, role, type_):
        return {"name": name, "role": role, "type": type_, "nullable": role != "Identifier"}

    value_domains = [
        {"name": "VD_D", "type": "Date", "setlist": ["2024-1-5", "2024-01-06"]},
        {"name": "VD_S", "type": "String", "setlist": ["A B", "x-y"]},
    ]
    structures = {
        "datasets": [
            {
                "name": name,
                "DataStructure": [
                    component("Id_1", "Identifier", "Integer"),
                    component("Id_2", "Identifier", value_domain),
                    component("Me_1", "Measure", "Integer"),
                ],
            }
            for name, value_domain in (("DS_1", "VD_D"), ("DS_2", "VD_S"))
        ]
    }
    analysis = semantic_analysis(
        "DS_r <- DS_1[pivot Id_2, Me_1]; DS_s <- DS_2[pivot Id_2, Me_1];", structures, value_domains
    )
    back = []
    for result, name in (("DS_r", "DS_1"), ("DS_s", "DS_2")):
        written = json.loads(analysis[result].to_json_datastructure())
        written["datasets"][0]["name"] = name
        back.append(written)
    assert [c["name"] for c in back[1]["datasets"][0]["DataStructure"]] == ["Id_1", "A B", "x-y"]
    back[1]["datasets"][0]["DataStructure"][2]["name"] = "'x-y'"
    ds_1 = tmp_path / "DS_1.csv"
    ds_1.write_text("Id_1,2024-01-05,2024-01-06\n1,1,2\n")
    ds_2 = tmp_path / "DS_2.csv"
    ds_2.write_text("Id_1,A B,x-y\n1,3,4\n")
    script = (
        "DS_a <- DS_1[calc Me_9 := '2024-01-05' + '2024-01-06'];"
        " DS_b <- DS_2[calc Me_9 := 'A B' * 'x-y'];"
    )
    result = run(script, back, {"DS_1": ds_1, "DS_2": ds_2}, use_duckdb=use_duckdb)
    assert result["DS_a"].data["Me_9"].tolist() == [3]
    assert result["DS_b"].data["Me_9"].tolist() == [12]
    assert list(result["DS_b"].components) == ["Id_1", "A B", "x-y", "Me_9"]
    validate_dataset(back[0], ds_1, use_duckdb=use_duckdb)


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize(
    "name",
    ['a"b', "'a\"b'", "a\x00b", "'a\x00b'", "a\rb", "'a\rb'"],
    ids=["double-quote", "quoted-double-quote", "nul", "quoted-nul", "cr", "quoted-cr"],
)
def test_GH_1349_57b(use_duckdb, name):
    """A Component name holding a double quote, a NUL or a carriage return is rejected by the
    structure schema (0-2-1-1), in run on both engines and in semantic_analysis, also between
    single quotes: the DuckDB engine cannot create the table of a name with a double quote or
    a NUL, and reads no row from a CSV file whose header holds a carriage return, so pandas
    alone would run it."""
    structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False},
                    {"name": name, "role": "Measure", "type": "Integer", "nullable": True},
                ],
            }
        ]
    }
    data = pd.DataFrame({"Id_1": [1], name.strip("'"): [1]})
    with pytest.raises(InputValidationException, match="0-2-1-1"):
        run("DS_r <- DS_1;", structures, {"DS_1": data}, use_duckdb=use_duckdb)
    with pytest.raises(InputValidationException, match="0-2-1-1"):
        semantic_analysis("DS_r <- DS_1;", structures)


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1352_39b(use_duckdb):
    """Extra of case 39: a Time_Period Scalar value is read without the space around it, as a
    loaded Time_Period and a Date Scalar value are, so "2020-01-15 " is the day 2020D15 and
    " 2020Q1" is 2020Q1. The pandas engine raised a raw ValueError and 2-1-19-2, and the DuckDB
    engine a raw ConversionException for " 2020Q1" (issue #1352)."""
    structures = {
        "datasets": [],
        "scalars": [
            {"name": "SC_1", "type": "Time_Period"},
            {"name": "SC_2", "type": "Time_Period"},
        ],
    }
    result = run(
        "SC_r1 <- SC_1; SC_r2 <- SC_2;",
        structures,
        {},
        scalar_values={"SC_1": "2020-01-15 ", "SC_2": " 2020Q1"},
        use_duckdb=use_duckdb,
    )
    assert result["SC_r1"].value == "2020D15"
    assert result["SC_r2"].value == "2020Q1"


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1352_13c(use_duckdb):
    """Extra of case 13: a Time_Period Scalar value and a Time_Period given by cast are held in
    one form, so the analytic max of an if over them orders the periods and gives 2020Q2. The
    pandas engine held the cast text as written beside the Scalar value in the loaders' form and
    ordered the texts, giving 2020Q1 (issue #1352)."""
    structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Integer", "role": "Measure", "nullable": True},
                ],
            }
        ],
        "scalars": [{"name": "SC_1", "type": "Time_Period"}],
    }
    script = (
        "DS_r <- DS_1[calc Me_2 := max((if Id_1 = 1 then SC_1 "
        'else cast("2020Q1", time_period)) over ())][keep Me_2];'
    )
    result = run(
        script,
        structures,
        {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [1, 2]})},
        scalar_values={"SC_1": "2020Q2"},
        use_duckdb=use_duckdb,
    )
    assert result["DS_r"].data["Me_2"].tolist() == ["2020Q2", "2020Q2"]
