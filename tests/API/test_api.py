import csv
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from pysdmx.io import get_datasets
from pysdmx.io.pd import PandasDataset
from pysdmx.model import (
    DataflowRef,
    Reference,
    Ruleset,
    Transformation,
    TransformationScheme,
    UserDefinedOperator,
)
from pysdmx.model.dataflow import Dataflow, Schema
from pysdmx.model.vtl import VtlDataflowMapping

import vtlengine.DataTypes as DataTypes
from tests.Helper import TestHelper
from vtlengine.API import (
    generate_sdmx,
    prettify,
    run,
    run_sdmx,
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
    to_vtl_json,
)
from vtlengine.DataTypes import Integer, String
from vtlengine.Exceptions import SemanticError
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


class SDMXTestsOutput(TestHelper):
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"

    ds_input_prefix = "DS_"

    warnings.filterwarnings("ignore", category=FutureWarning)


input_vtl_params_OK = [
    (filepath_VTL / "2.vtl", "DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;"),
    (
        "DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;",
        "DS_r := DS_1 + DS_2; DS_r2 <- DS_1 + DS_r;",
    ),
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
        "The given json does not follow the schema.",
    ),
    (2, "Invalid vd file. Input is not a Path object"),
    (
        {"setlist": ["AT", "BE", "CY"], "type": "String"},
        "The given json does not follow the schema.",
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
]

load_datasets_with_data_and_wrong_inputs = [
    (
        filepath_csv / "DS_1.csv",
        filepath_csv / "DS_1.csv",
        "Invalid datastructure. Must have .json extension",
    ),
    (
        filepath_json / "DS_1.json",
        filepath_json / "DS_2.json",
        "Not found dataset DS_2.json",
    ),
    (2, 2, "Invalid datastructure. Input must be a dict or Path object"),
]

ext_params_OK = [(filepath_sql / "1.json")]

ext_params_wrong = [
    (
        filepath_json / "DS_1.json",
        "The given json does not follow the schema.",
    ),
    (5, "Input invalid. Input must be a json file."),
    (filepath_sql / "6.sql", "Input invalid. Input does not exist"),
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
    )
]

params_schema = [(filepath_json / "DS_Schema.json")]

param_id_null = [((filepath_json / "DS_ID_null.json"), "Identifier Id_1 cannot be nullable")]

param_wrong_role = [((filepath_json / "DS_Role_wrong.json"), "0-1-1-13")]

param_wrong_data_type = [((filepath_json / "DS_wrong_datatype.json"), "0-1-1-13")]

param_viral_attr = [((filepath_json / "DS_Viral_attr.json"), "0-1-1-13")]

params_run_sdmx = [
    (
        (filepath_sdmx_input / "gen_all_minimal.xml"),
        (filepath_sdmx_input / "metadata_minimal.xml"),
    ),
    (
        (filepath_sdmx_input / "str_all_minimal.xml"),
        (filepath_sdmx_input / "metadata_minimal.xml"),
    ),
]

params_run_sdmx_with_mappings = [
    (
        (filepath_sdmx_input / "str_all_minimal_df.xml"),
        (filepath_sdmx_input / "metadata_minimal_df.xml"),
        None,
    ),
    (
        (filepath_sdmx_input / "str_all_minimal_df.xml"),
        (filepath_sdmx_input / "metadata_minimal_df.xml"),
        {"Dataflow=MD:TEST_DF(1.0)": "DS_1"},
    ),
    (
        (filepath_sdmx_input / "str_all_minimal_df.xml"),
        (filepath_sdmx_input / "metadata_minimal_df.xml"),
        VtlDataflowMapping(
            dataflow="urn:sdmx:org.sdmx.infomodel.datastructure.Dataflow=MD:TEST_DF(1.0)",
            dataflow_alias="DS_1",
            id="VTL_MAP_1",
        ),
    ),
    (
        (filepath_sdmx_input / "str_all_minimal_df.xml"),
        (filepath_sdmx_input / "metadata_minimal_df.xml"),
        VtlDataflowMapping(
            dataflow=Reference(
                sdmx_type="Dataflow",
                agency="MD",
                id="TEST_DF",
                version="1.0",
            ),
            dataflow_alias="DS_1",
            id="VTL_MAP_2",
        ),
    ),
    (
        (filepath_sdmx_input / "str_all_minimal_df.xml"),
        (filepath_sdmx_input / "metadata_minimal_df.xml"),
        VtlDataflowMapping(
            dataflow=DataflowRef(
                agency="MD",
                id="TEST_DF",
                version="1.0",
            ),
            dataflow_alias="DS_1",
            id="VTL_MAP_3",
        ),
    ),
    (
        (filepath_sdmx_input / "str_all_minimal_df.xml"),
        (filepath_sdmx_input / "metadata_minimal_df.xml"),
        VtlDataflowMapping(
            dataflow=Dataflow(
                id="TEST_DF",
                agency="MD",
                version="1.0",
            ),
            dataflow_alias="DS_1",
            id="VTL_MAP_4",
        ),
    ),
]

params_run_sdmx_errors = [
    (
        [
            PandasDataset(
                structure=Schema(id="DS1", components=[], agency="BIS", context="datastructure"),
                data=pd.DataFrame(),
            ),
            PandasDataset(
                structure=Schema(id="DS2", components=[], agency="BIS", context="datastructure"),
                data=pd.DataFrame(),
            ),
        ],
        None,
        SemanticError,
        "0-1-3-3",
    ),
    (
        [
            PandasDataset(
                structure=Schema(
                    id="BIS_DER", components=[], agency="BIS", context="datastructure"
                ),
                data=pd.DataFrame(),
            )
        ],
        42,
        TypeError,
        "Expected dict or VtlDataflowMapping type for mappings.",
    ),
    (
        [
            PandasDataset(
                structure=Schema(
                    id="BIS_DER", components=[], agency="BIS", context="datastructure"
                ),
                data=pd.DataFrame(),
            )
        ],
        VtlDataflowMapping(
            dataflow=123,
            dataflow_alias="ALIAS",
            id="Test",
        ),
        TypeError,
        "Expected str, Reference, DataflowRef or Dataflow type for dataflow in VtlDataflowMapping.",
    ),
]
params_to_vtl_json = [
    (
        (filepath_sdmx_input / "str_all_minimal.xml"),
        (filepath_sdmx_input / "metadata_minimal.xml"),
        (filepath_sdmx_output / "vtl_datastructure_str_all.json"),
    )
]

params_2_1_str_sp = [
    (
        "1-1",
        (filepath_sdmx_input / "str_all_minimal.xml"),
        (filepath_sdmx_input / "metadata_minimal.xml"),
    )
]

params_2_1_gen_str = [
    (
        "1-2",
        (filepath_sdmx_input / "str_all_minimal.xml"),
        (filepath_sdmx_input / "metadata_minimal.xml"),
    )
]

params_exception_vtl_to_json = [((filepath_sdmx_input / "str_all_minimal.xml"), "0-1-3-2")]

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
        "On Dataset DS_1 loading: Component Id_1 is missing in Datapoints.",
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
        "On Dataset DS_1 loading: Component Id_1 is missing in Datapoints.",
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
    exception_code = "0-1-1-10"

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

    with pytest.raises(SemanticError) as context:
        run(script=script, data_structures=data_structures, datapoints=datapoints)
    result = exception_code == str(context.value.args[1])
    if result is False:
        print(f"\n{exception_code} != {context.value.args[1]}")
    assert result


def test_mandatory_me_error():
    exception_code = "0-1-1-10"

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

    with pytest.raises(SemanticError) as context:
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


@pytest.mark.parametrize("data, structure", params_run_sdmx)
def test_run_sdmx_function(data, structure):
    script = "DS_r := BIS_DER [calc Me_4 := OBS_VALUE];"
    datasets = get_datasets(data, structure)
    result = run_sdmx(script, datasets, return_only_persistent=False)
    assert isinstance(result, dict)
    assert all(isinstance(k, str) and isinstance(v, Dataset) for k, v in result.items())
    assert isinstance(result["DS_r"].data, pd.DataFrame)


@pytest.mark.parametrize("data, structure, mappings", params_run_sdmx_with_mappings)
def test_run_sdmx_function_with_mappings(data, structure, mappings):
    script = "DS_r := DS_1 [calc Me_4 := OBS_VALUE];"
    datasets = get_datasets(data, structure)
    result = run_sdmx(script, datasets, mappings=mappings, return_only_persistent=False)
    assert isinstance(result, dict)
    assert all(isinstance(k, str) and isinstance(v, Dataset) for k, v in result.items())
    assert isinstance(result["DS_r"].data, pd.DataFrame)


@pytest.mark.parametrize("datasets, mappings, expected_exception, match", params_run_sdmx_errors)
def test_run_sdmx_errors_with_mappings(datasets, mappings, expected_exception, match):
    script = "DS_r := BIS_DER [calc Me_4 := OBS_VALUE];"
    with pytest.raises(expected_exception, match=match):
        run_sdmx(script, datasets, mappings=mappings)


@pytest.mark.parametrize("data, structure, path_reference", params_to_vtl_json)
def test_to_vtl_json_function(data, structure, path_reference):
    datasets = get_datasets(data, structure)
    result = to_vtl_json(datasets[0].structure, dataset_name="BIS_DER")
    with open(path_reference, "r") as file:
        reference = json.load(file)
    assert result == reference


@pytest.mark.parametrize("code, data, structure", params_2_1_str_sp)
def test_run_sdmx_2_1_str_sp(code, data, structure):
    datasets = get_datasets(data, structure)
    result = run_sdmx(
        "DS_r := BIS_DER [calc Me_4 := OBS_VALUE];", datasets, return_only_persistent=False
    )
    reference = SDMXTestsOutput.LoadOutputs(code, ["DS_r"])
    assert result == reference


@pytest.mark.parametrize("code, data, structure", params_2_1_gen_str)
def test_run_sdmx_2_1_gen_all(code, data, structure):
    datasets = get_datasets(data, structure)
    result = run_sdmx(
        "DS_r := BIS_DER [calc Me_4 := OBS_VALUE];", datasets, return_only_persistent=False
    )
    reference = SDMXTestsOutput.LoadOutputs(code, ["DS_r"])
    assert result == reference


@pytest.mark.parametrize("data, error_code", params_exception_vtl_to_json)
def test_to_vtl_json_exception(data, error_code):
    datasets = get_datasets(data)
    with pytest.raises(SemanticError, match=error_code):
        run_sdmx("DS_r := BIS_DER [calc Me_4 := OBS_VALUE];", datasets)


def test_ts_without_udo_or_rs():
    script = "DS_r := DS_1 + DS_2;"
    ts = generate_sdmx(script, agency_id="MD", id="TestID")

    assert isinstance(ts, TransformationScheme)
    assert ts.id == "TS1"
    assert ts.agency == "MD"
    assert ts.version == "1.0"
    assert ts.name == "TransformationScheme TestID"
    assert len(ts.items) == 1
    transformation = ts.items[0]
    assert transformation.is_persistent is False


def test_ts_with_udo():
    script = """
    define operator suma (ds1 dataset, ds2 dataset)
            returns dataset is
            ds1 + ds2
    end operator;
    DS_r := suma(ds1, ds2);
    """
    ts = generate_sdmx(script, agency_id="MD", id="TestID")
    assert isinstance(ts, TransformationScheme)
    assert len(ts.items) == 1
    udo_scheme = ts.user_defined_operator_schemes[0]
    assert udo_scheme.id == "UDS1"
    assert udo_scheme.name == "UserDefinedOperatorScheme TestID-UDS"
    assert len(udo_scheme.items) == 1
    udo = udo_scheme.items[0]
    assert isinstance(udo, UserDefinedOperator)
    assert udo.id == "UDO1"


def test_ts_with_dp_ruleset():
    script = """
    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
        FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
        sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
        sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1
    end datapoint ruleset;
    DS_r := check_datapoint (BOP, signValidation);
    """
    ts = generate_sdmx(script, agency_id="MD", id="TestID")
    assert isinstance(ts, TransformationScheme)
    assert hasattr(ts, "ruleset_schemes")
    rs_scheme = ts.ruleset_schemes[0]
    assert rs_scheme.id == "RS1"
    assert rs_scheme.name == "RulesetScheme TestID-RS"
    assert len(rs_scheme.items) == 1
    ruleset = rs_scheme.items[0]
    assert isinstance(ruleset, Ruleset)
    assert ruleset.id == "R1"
    assert ruleset.ruleset_type == "datapoint"


def test_ts_with_hierarchical_ruleset():
    script = """
        define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

        DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset);
        """
    ts = generate_sdmx(script, agency_id="MD", id="TestID")
    assert isinstance(ts, TransformationScheme)
    assert hasattr(ts, "ruleset_schemes")
    rs_scheme = ts.ruleset_schemes[0]
    assert rs_scheme.id == "RS1"
    assert rs_scheme.name == "RulesetScheme TestID-RS"
    assert len(rs_scheme.items) == 1
    ruleset = rs_scheme.items[0]
    assert isinstance(ruleset, Ruleset)
    assert ruleset.id == "R1"
    assert ruleset.ruleset_type == "hierarchical"
    assert ruleset.ruleset_definition == (
        "define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is "
        'B = C - D errorcode "Balance (credit-debit)" errorlevel 4; N = A - L errorcode "Net (assets-liabilities)" errorlevel 4 end hierarchical ruleset;'
    )


def test_ts_with_2_rulesets():
    script = filepath_VTL / "validations.vtl"
    ts = generate_sdmx(script, agency_id="MD", id="TestID")
    assert isinstance(ts, TransformationScheme)
    rs_scheme = ts.ruleset_schemes[0]
    assert rs_scheme.id == "RS1"
    assert len(rs_scheme.items) == 2
    assert isinstance(rs_scheme.items[0], Ruleset)
    assert rs_scheme.items[0].ruleset_type == "datapoint"


def test_ts_with_ruleset_and_udo():
    script = """
    define operator suma (ds1 dataset, ds2 dataset)
            returns dataset is
            ds1 + ds2
    end operator;
    DS_r := suma(ds1, ds2);

    define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

    DS_r2 := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset);
    """
    ts = generate_sdmx(script, agency_id="MD", id="TestID")

    # Validate TransformationScheme
    assert isinstance(ts, TransformationScheme)

    # Validate UDO scheme
    assert hasattr(ts, "user_defined_operator_schemes")
    assert len(ts.user_defined_operator_schemes) == 1
    udo_scheme = ts.user_defined_operator_schemes[0]
    assert udo_scheme.id == "UDS1"
    assert len(udo_scheme.items) == 1
    assert isinstance(udo_scheme.items[0], UserDefinedOperator)

    # Validate Ruleset scheme
    assert hasattr(ts, "ruleset_schemes")
    rs_scheme = ts.ruleset_schemes[0]
    assert rs_scheme.id == "RS1"
    assert len(rs_scheme.items) == 1
    assert isinstance(rs_scheme.items[0], Ruleset)
    assert rs_scheme.items[0].ruleset_type == "hierarchical"
    ruleset = rs_scheme.items[0]
    assert isinstance(ruleset, Ruleset)
    assert ruleset.id == "R1"
    assert ruleset.ruleset_type == "hierarchical"
    assert ruleset.ruleset_definition == (
        "define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is "
        'B = C - D errorcode "Balance (credit-debit)" errorlevel 4; N = A - L errorcode "Net (assets-liabilities)" errorlevel 4 end hierarchical ruleset;'
    )


def test_check_script_with_string_input():
    script = "DS_r := DS_1 + DS_2;"
    result = _check_script(script)
    assert result == script


def test_check_script_invalid_input_type():
    with pytest.raises(
        Exception,
        match="invalid script format type: int. Input must be a string, TransformationScheme or Path object",
    ):
        _check_script(12345)


def test_generate_sdmx_and_check_script():
    script = """
    define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
    end hierarchical ruleset;
    define operator suma (ds1 dataset, ds2 dataset)
            returns dataset is
            ds1 + ds2
    end operator;
    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset);
    DS_r2 := suma(ds1, ds2);
    """
    ts = generate_sdmx(script, agency_id="MD", id="TestID")
    assert isinstance(ts, TransformationScheme)
    assert hasattr(ts, "user_defined_operator_schemes")
    assert len(ts.user_defined_operator_schemes) == 1
    udo = ts.user_defined_operator_schemes[0]
    assert isinstance(udo.items[0], UserDefinedOperator)
    assert hasattr(ts, "ruleset_schemes")
    rs = ts.ruleset_schemes[0]
    assert isinstance(rs.items[0], Ruleset)
    assert rs.items[0].ruleset_type == "hierarchical"
    assert rs.items[0].ruleset_scope == "variable"
    regenerated_script = _check_script(ts)
    assert prettify(script) == prettify(regenerated_script)


def test_generate_sdmx_and_check_script_with_valuedomain():
    script = """
    define hierarchical ruleset sectorsHierarchy (valuedomain rule abstract) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
    end hierarchical ruleset;
    define operator suma (ds1 dataset, ds2 dataset)
            returns dataset is
            ds1 + ds2
    end operator;
    sectors_hier_val_unf := check_hierarchy(DS_1, sectorsHierarchy rule Id_2 non_zero);
    DS_r2 := suma(ds1, ds2);
    """
    ts = generate_sdmx(script, agency_id="MD", id="TestID")
    assert isinstance(ts, TransformationScheme)
    assert hasattr(ts, "user_defined_operator_schemes")
    assert len(ts.user_defined_operator_schemes) == 1
    udo = ts.user_defined_operator_schemes[0]
    assert isinstance(udo.items[0], UserDefinedOperator)
    assert hasattr(ts, "ruleset_schemes")
    rs = ts.ruleset_schemes[0]
    assert isinstance(rs.items[0], Ruleset)
    assert rs.items[0].ruleset_type == "hierarchical"
    assert rs.items[0].ruleset_scope == "valuedomain"
    regenerated_script = _check_script(ts)
    assert prettify(script) == prettify(regenerated_script)


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
    }
    assert run_result == reference
    ds_csv = output_folder / "DS_r.csv"
    sc_csv = output_folder / "Sc_r.csv"
    assert ds_csv.exists()
    assert sc_csv.exists()
    df = pd.read_csv(ds_csv)
    assert list(df.columns) == ["Id_1", "Me_1"]
    assert df.loc[0, "Id_1"] == 2
    assert df.loc[0, "Me_1"] == 20
    with open(sc_csv, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0][0] == str(reference["Sc_r"].value)
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
    sc_csv = output_folder / "Sc_r.csv"
    assert ds_csv.exists()
    assert sc_csv.exists()
    df = pd.read_csv(ds_csv)
    assert list(df.columns) == ["Id_1", "Me_1"]
    assert df.loc[0, "Id_1"] == 2
    assert df.loc[0, "Me_1"] == 20
    with open(sc_csv, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0] == [] or rows[0] == [""]


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
    with pytest.raises(Exception, match="The given json does not follow the schema."):
        _validate_json(vd_data, vd_schema)


@pytest.mark.parametrize("path_schema, path_sql", params_invalid_sql)
def test_attempt_to_validate_invalid_sql(path_schema, path_sql):
    with open(path_sql, "r") as f:
        ext_routine_data = json.load(f)
    with open(path_schema, "r") as f:
        ext_routine_schema = json.load(f)
    try:
        _validate_json(ext_routine_data, ext_routine_schema)
    except Exception:
        with pytest.raises(Exception, match="The given json does not follow the schema."):
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
      DS_r3 <- eval(SQL_3(DS_1) language "sqlite" returns dataset {identifier<integer> Id_1, measure<number> Me_1});
      DS_r4 <- eval(SQL_4(DS_1) language "sqlite" returns dataset {identifier<integer> Id_1, measure<number> Me_1});
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
          DS_r <- DS_1 [ calc Me_2:= Me_1 in Countries];
          DS_r2 <- DS_1 [ calc Me_2:= Me_1 in Countries_EU_Sample];
          DS_r3 <- eval(SQL_3(DS_1) language "sqlite" returns dataset {identifier<integer> Id_1, measure<number> Me_1});
          DS_r4 <- eval(SQL_4(DS_1) language "sqlite" returns dataset {identifier<integer> Id_1, measure<number> Me_1});
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
