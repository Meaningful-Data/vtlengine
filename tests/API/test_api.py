import csv
import json
from pathlib import Path

import pandas as pd
import pytest
from pysdmx.model import (
    Transformation,
    TransformationScheme,
)

import vtlengine.DataTypes as DataTypes
from vtlengine.API import (
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

param_wrong_role = [((filepath_json / "DS_Role_wrong.json"), "0-1-1-13")]

param_wrong_data_type = [((filepath_json / "DS_wrong_datatype.json"), "0-1-1-13")]

param_viral_attr = [((filepath_json / "DS_Viral_attr.json"), "0-1-1-13")]

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
        "0-3-1-5",
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
        "0-3-1-5",
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
                role=Role.ATTRIBUTE,
                nullable=True,
            ),
        },
        data=None,
    )
    assert "DS_Schema" in datasets
    assert datasets["DS_Schema"] == reference


@pytest.mark.parametrize("ds_r, error_message", param_id_null)
def test_load_data_structure_with_null_id(ds_r, error_message):
    with pytest.raises(ValueError, match=error_message):
        load_datasets(ds_r)


@pytest.mark.parametrize("ds_r, error_code", param_wrong_role)
def test_load_data_structure_with_wrong_role(ds_r, error_code):
    with pytest.raises(SemanticError, match=error_code):
        load_datasets(ds_r)


@pytest.mark.parametrize("ds_r, error_code", param_wrong_data_type)
def test_load_data_structure_with_wrong_data_type(ds_r, error_code):
    with pytest.raises(SemanticError, match=error_code):
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


@pytest.mark.parametrize("data_structures, datapoints", params_run_with_scalars)
def test_run_with_scalars(data_structures, datapoints, tmp_path):
    script = """
        DS_r <- DS_3[filter Me_1 = sc_1];
        DS_r2 <- DS_3[sub Id_1 = sc_1];
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
            data=pd.DataFrame({"Id_1": [2], "Me_1": [20]}),
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
            data=pd.DataFrame({"Me_1": []}),
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
        DS_r2 <- DS_3[sub Id_1 = sc_1];
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
            data=pd.DataFrame({"Id_1": [2], "Me_1": [20]}),
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
            data=pd.DataFrame({"Me_1": []}),
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

    with pytest.raises(SemanticError, match="0-1-1-13") as e:
        run(
            script=script,
            data_structures=data_structures,
            datapoints=[],
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
    _validate_json(vd_data, vd_schema)
    _validate_json(ext_routine_data, ext_routine_schema)


@pytest.mark.parametrize("path_schema, path_vd", params_invalid_vd)
def test_attempt_to_validate_invalid_vd(path_schema, path_vd):
    with open(path_vd, "r") as f:
        vd_data = json.load(f)
    with open(path_schema, "r") as f:
        vd_schema = json.load(f)
    with pytest.raises(InputValidationException, match="0-2-1-1"):
        _validate_json(vd_data, vd_schema)


@pytest.mark.parametrize("path_schema, path_sql", params_invalid_sql)
def test_attempt_to_validate_invalid_sql(path_schema, path_sql):
    with open(path_sql, "r") as f:
        ext_routine_data = json.load(f)
    with open(path_schema, "r") as f:
        ext_routine_schema = json.load(f)
    try:
        _validate_json(ext_routine_data, ext_routine_schema)
    except InputValidationException:
        with pytest.raises(InputValidationException, match="0-2-1-1"):
            _validate_json(ext_routine_data, ext_routine_schema)
        return
    query = ext_routine_data.get("query")
    name = ext_routine_data.get("name", "test_routine")

    with pytest.raises(Exception, match="Invalid SQL query in external routine"):
        ExternalRoutine.from_sql_query(name, query)


def test_with_multiple_vd_and_ext_routines():
    script = """
      DS_r <- DS_1 [ calc Me_2:= Me_1 in Countries];
      DS_r2 <- DS_1 [ calc Me_2:= Me_1 in Countries_EU_Sample];
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

    with pytest.raises(SemanticError) as context:
        run(
            script=script,
            data_structures=data_structures,
            datapoints=datapoints,
            value_domains=value_domains,
            external_routines=external_routines,
        )
    assert context.value.args[1] == "1-1-1-1"


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
          DS_r <- DS_1 [ calc Me_2:= Me_1 in Countries];
          DS_r2 <- DS_1 [ calc Me_2:= Me_1 in Countries_EU_Sample];
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

    with pytest.raises(SemanticError) as context:
        semantic_analysis(
            script=script,
            data_structures=data_structures,
            value_domains=value_domains,
            external_routines=external_routines,
        )
    assert context.value.args[1] == "1-1-1-1"


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


@pytest.mark.parametrize("ds_input, dp_input, is_valid, message", params_validate_ds)
def test_validate_dataset(ds_input, dp_input, is_valid, message):
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
        validate_dataset(ds_data, dp_input)
    else:
        with pytest.raises(Exception, match=message):
            validate_dataset(ds_data, dp_input)
