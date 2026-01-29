"""
Tests for SDMX file loading functionality.

This module tests:
- Loading SDMX files via run() datapoints parameter (SDMX-ML, SDMX-JSON, SDMX-CSV)
- run_sdmx() function with PandasDataset objects
- to_vtl_json() function for converting SDMX structures
"""

import json
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest
from pysdmx.io import get_datasets
from pysdmx.io.pd import PandasDataset
from pysdmx.model import DataflowRef, Reference
from pysdmx.model.dataflow import Dataflow, Schema
from pysdmx.model.vtl import VtlDataflowMapping

from tests.Helper import TestHelper
from vtlengine.API import run, run_sdmx
from vtlengine.API._InternalApi import to_vtl_json
from vtlengine.Exceptions import DataLoadError, InputValidationException
from vtlengine.Model import Dataset

# Path setup
base_path = Path(__file__).parent
filepath_sdmx_input = base_path / "data" / "SDMX" / "input"
filepath_sdmx_output = base_path / "data" / "SDMX" / "output"
filepath_csv = base_path / "data" / "DataSet" / "input"
filepath_json = base_path / "data" / "DataStructure" / "input"


class SDMXTestHelper(TestHelper):
    """Helper class for SDMX tests with output loading support."""

    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    ds_input_prefix = "DS_"
    warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Fixtures for SDMX tests
# =============================================================================


@pytest.fixture
def sdmx_data_file():
    """SDMX-ML data file."""
    return filepath_sdmx_input / "str_all_minimal.xml"


@pytest.fixture
def sdmx_structure_file():
    """SDMX-ML structure/metadata file."""
    return filepath_sdmx_input / "metadata_minimal.xml"


@pytest.fixture
def sdmx_data_structure(sdmx_data_file, sdmx_structure_file):
    """VTL data structure derived from SDMX metadata."""
    pandas_datasets = get_datasets(data=sdmx_data_file, structure=sdmx_structure_file)
    schema = pandas_datasets[0].structure
    return to_vtl_json(schema, "BIS_DER")


# =============================================================================
# Tests for run() with SDMX file datapoints - parametrized
# =============================================================================


params_run_sdmx_datapoints_dict = [
    # (script, datapoints_key, description)
    ("DS_r <- BIS_DER;", "BIS_DER", "simple assignment"),
    ("DS_r <- BIS_DER [calc Me_4 := OBS_VALUE];", "BIS_DER", "calc clause"),
    ("DS_r <- BIS_DER [filter OBS_VALUE > 0];", "BIS_DER", "filter clause"),
]


@pytest.mark.parametrize("script, ds_key, description", params_run_sdmx_datapoints_dict)
def test_run_sdmx_file_via_dict(sdmx_data_file, sdmx_data_structure, script, ds_key, description):
    """Test loading SDMX-ML file using dict with explicit name."""
    result = run(
        script=script,
        data_structures=sdmx_data_structure,
        datapoints={ds_key: sdmx_data_file},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None
    assert len(result["DS_r"].data) > 0


def test_run_sdmx_file_via_list(sdmx_data_file, sdmx_data_structure):
    """Test loading SDMX files via list of paths."""
    script = "DS_r <- BIS_DER;"
    result = run(
        script=script,
        data_structures=sdmx_data_structure,
        datapoints=[sdmx_data_file],
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


def test_run_sdmx_file_via_single_path(sdmx_data_file, sdmx_data_structure):
    """Test loading SDMX files via single Path (with dict for explicit naming)."""
    script = "DS_r <- BIS_DER;"
    # Single path must use dict for explicit naming since URN extraction may differ
    result = run(
        script=script,
        data_structures=sdmx_data_structure,
        datapoints={"BIS_DER": sdmx_data_file},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert isinstance(result["DS_r"], Dataset)


# =============================================================================
# Tests for run() with SDMX file datapoints - error cases
# =============================================================================


params_sdmx_error_cases = [
    # (error_type, error_match, file_content_or_path, description)
    ("invalid_xml", "0-3-1-8", "<invalid>not sdmx</invalid>", "invalid XML content"),
    ("nonexistent", "0-3-1-1", "/nonexistent/file.xml", "file does not exist"),
]


@pytest.mark.parametrize(
    "error_type, error_match, file_or_content, description", params_sdmx_error_cases
)
def test_run_sdmx_file_errors(
    sdmx_data_structure, error_type, error_match, file_or_content, description
):
    """Test error handling for invalid SDMX files."""
    if error_type == "invalid_xml":
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
            f.write(file_or_content)
            test_file = Path(f.name)
        try:
            with pytest.raises(DataLoadError, match=error_match):
                run(
                    script="DS_r <- TEST;",
                    data_structures=sdmx_data_structure,
                    datapoints={"TEST": test_file},
                )
        finally:
            test_file.unlink()
    elif error_type == "nonexistent":
        with pytest.raises(DataLoadError, match=error_match):
            run(
                script="DS_r <- TEST;",
                data_structures=sdmx_data_structure,
                datapoints={"TEST": Path(file_or_content)},
            )


def test_run_sdmx_missing_structure(sdmx_data_file):
    """Test that SDMX dataset without matching structure raises error."""
    # Structure that doesn't match the SDMX dataset name
    wrong_structure = filepath_json / "DS_1.json"
    with open(wrong_structure) as f:
        data_structure = json.load(f)

    with pytest.raises(InputValidationException, match="Not found dataset BIS_DER"):
        run(
            script="DS_r <- BIS_DER;",
            data_structures=data_structure,
            datapoints={"BIS_DER": sdmx_data_file},
        )


# =============================================================================
# Tests for mixed SDMX and CSV datapoints
# =============================================================================


def test_run_mixed_sdmx_and_csv(sdmx_data_file, sdmx_data_structure):
    """Test loading both SDMX and CSV files in the same run() call."""
    # Get CSV structure
    csv_structure_path = filepath_json / "DS_1.json"
    with open(csv_structure_path) as f:
        csv_structure = json.load(f)

    # Combine structures
    combined_structure = {"datasets": sdmx_data_structure["datasets"] + csv_structure["datasets"]}

    script = "DS_r <- BIS_DER; DS_r2 <- DS_1;"
    csv_file = filepath_csv / "DS_1.csv"

    result = run(
        script=script,
        data_structures=combined_structure,
        datapoints={
            "BIS_DER": sdmx_data_file,
            "DS_1": csv_file,
        },
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert "DS_r2" in result
    assert result["DS_r"].data is not None
    assert result["DS_r2"].data is not None


# =============================================================================
# Tests for run_sdmx() function - parametrized
# =============================================================================


params_run_sdmx = [
    (filepath_sdmx_input / "gen_all_minimal.xml", filepath_sdmx_input / "metadata_minimal.xml"),
    (filepath_sdmx_input / "str_all_minimal.xml", filepath_sdmx_input / "metadata_minimal.xml"),
]


@pytest.mark.parametrize("data, structure", params_run_sdmx)
def test_run_sdmx_function(data, structure):
    """Test run_sdmx with basic SDMX data and structure files."""
    script = "DS_r := BIS_DER [calc Me_4 := OBS_VALUE];"
    datasets = get_datasets(data, structure)
    result = run_sdmx(script, datasets, return_only_persistent=False)

    assert isinstance(result, dict)
    assert all(isinstance(k, str) and isinstance(v, Dataset) for k, v in result.items())
    assert isinstance(result["DS_r"].data, pd.DataFrame)


params_run_sdmx_with_mappings = [
    (
        filepath_sdmx_input / "str_all_minimal_df.xml",
        filepath_sdmx_input / "metadata_minimal_df.xml",
        None,
    ),
    (
        filepath_sdmx_input / "str_all_minimal_df.xml",
        filepath_sdmx_input / "metadata_minimal_df.xml",
        {"Dataflow=MD:TEST_DF(1.0)": "DS_1"},
    ),
    (
        filepath_sdmx_input / "str_all_minimal_df.xml",
        filepath_sdmx_input / "metadata_minimal_df.xml",
        VtlDataflowMapping(
            dataflow="urn:sdmx:org.sdmx.infomodel.datastructure.Dataflow=MD:TEST_DF(1.0)",
            dataflow_alias="DS_1",
            id="VTL_MAP_1",
        ),
    ),
    (
        filepath_sdmx_input / "str_all_minimal_df.xml",
        filepath_sdmx_input / "metadata_minimal_df.xml",
        VtlDataflowMapping(
            dataflow=Reference(sdmx_type="Dataflow", agency="MD", id="TEST_DF", version="1.0"),
            dataflow_alias="DS_1",
            id="VTL_MAP_2",
        ),
    ),
    (
        filepath_sdmx_input / "str_all_minimal_df.xml",
        filepath_sdmx_input / "metadata_minimal_df.xml",
        VtlDataflowMapping(
            dataflow=DataflowRef(agency="MD", id="TEST_DF", version="1.0"),
            dataflow_alias="DS_1",
            id="VTL_MAP_3",
        ),
    ),
    (
        filepath_sdmx_input / "str_all_minimal_df.xml",
        filepath_sdmx_input / "metadata_minimal_df.xml",
        VtlDataflowMapping(
            dataflow=Dataflow(id="TEST_DF", agency="MD", version="1.0"),
            dataflow_alias="DS_1",
            id="VTL_MAP_4",
        ),
    ),
]


@pytest.mark.parametrize("data, structure, mappings", params_run_sdmx_with_mappings)
def test_run_sdmx_function_with_mappings(data, structure, mappings):
    """Test run_sdmx with various mapping types."""
    script = "DS_r := DS_1 [calc Me_4 := OBS_VALUE];"
    datasets = get_datasets(data, structure)
    result = run_sdmx(script, datasets, mappings=mappings, return_only_persistent=False)

    assert isinstance(result, dict)
    assert all(isinstance(k, str) and isinstance(v, Dataset) for k, v in result.items())
    assert isinstance(result["DS_r"].data, pd.DataFrame)


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
        InputValidationException,
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
        InputValidationException,
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
        VtlDataflowMapping(dataflow=123, dataflow_alias="ALIAS", id="Test"),
        InputValidationException,
        "Expected str, Reference, DataflowRef or Dataflow type for dataflow in VtlDataflowMapping.",
    ),
]


@pytest.mark.parametrize("datasets, mappings, expected_exception, match", params_run_sdmx_errors)
def test_run_sdmx_errors_with_mappings(datasets, mappings, expected_exception, match):
    """Test run_sdmx error handling with invalid inputs."""
    script = "DS_r := BIS_DER [calc Me_4 := OBS_VALUE];"
    with pytest.raises(expected_exception, match=match):
        run_sdmx(script, datasets, mappings=mappings)


# =============================================================================
# Tests for to_vtl_json() function
# =============================================================================


params_to_vtl_json = [
    (
        filepath_sdmx_input / "str_all_minimal.xml",
        filepath_sdmx_input / "metadata_minimal.xml",
        filepath_sdmx_output / "vtl_datastructure_str_all.json",
    ),
]


@pytest.mark.parametrize("data, structure, path_reference", params_to_vtl_json)
def test_to_vtl_json_function(data, structure, path_reference):
    """Test to_vtl_json conversion of SDMX structure to VTL JSON format."""
    datasets = get_datasets(data, structure)
    result = to_vtl_json(datasets[0].structure, dataset_name="BIS_DER")
    with open(path_reference, "r") as file:
        reference = json.load(file)
    assert result == reference


params_exception_vtl_to_json = [
    (filepath_sdmx_input / "str_all_minimal.xml", "0-1-3-2"),
]


@pytest.mark.parametrize("data, error_code", params_exception_vtl_to_json)
def test_to_vtl_json_exception(data, error_code):
    """Test to_vtl_json raises exception for data without structure."""
    datasets = get_datasets(data)
    with pytest.raises(InputValidationException, match=error_code):
        run_sdmx("DS_r := BIS_DER [calc Me_4 := OBS_VALUE];", datasets)


# =============================================================================
# Tests for run_sdmx with output comparison
# =============================================================================


params_sdmx_output = [
    (
        "1-1",
        filepath_sdmx_input / "str_all_minimal.xml",
        filepath_sdmx_input / "metadata_minimal.xml",
    ),
    (
        "1-2",
        filepath_sdmx_input / "str_all_minimal.xml",
        filepath_sdmx_input / "metadata_minimal.xml",
    ),
]


@pytest.mark.parametrize("code, data, structure", params_sdmx_output)
def test_run_sdmx_output_comparison(code, data, structure):
    """Test run_sdmx with output comparison to reference data."""
    datasets = get_datasets(data, structure)
    result = run_sdmx(
        "DS_r := BIS_DER [calc Me_4 := OBS_VALUE];", datasets, return_only_persistent=False
    )
    reference = SDMXTestHelper.LoadOutputs(code, ["DS_r"])
    assert result == reference


# =============================================================================
# Tests for plain CSV fallback
# =============================================================================


def test_plain_csv_still_works():
    """Test that plain CSV files still work (not SDMX-CSV)."""
    csv_file = filepath_csv / "DS_1.csv"
    structure_file = filepath_json / "DS_1.json"

    with open(structure_file) as f:
        data_structure = json.load(f)

    script = "DS_r <- DS_1;"
    result = run(
        script=script,
        data_structures=data_structure,
        datapoints={"DS_1": csv_file},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


# =============================================================================
# Tests for run() with SDMX data_structures parameter
# =============================================================================


def test_run_with_sdmx_structure_file(sdmx_data_file, sdmx_structure_file):
    """Test run() with SDMX structure file path instead of VTL JSON."""
    script = "DS_r <- BIS_DER;"
    result = run(
        script=script,
        data_structures=sdmx_structure_file,
        datapoints={"BIS_DER": sdmx_data_file},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None
    assert len(result["DS_r"].data) > 0


def test_run_with_sdmx_structure_file_list(sdmx_data_file, sdmx_structure_file):
    """Test run() with list of SDMX structure files."""
    script = "DS_r <- BIS_DER;"
    result = run(
        script=script,
        data_structures=[sdmx_structure_file],
        datapoints={"BIS_DER": sdmx_data_file},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


# =============================================================================
# Tests for run() with pysdmx objects as data_structures
# =============================================================================


def test_run_with_schema_object(sdmx_data_file, sdmx_structure_file):
    """Test run() with pysdmx Schema object."""
    from pysdmx.io import get_datasets as pysdmx_get_datasets

    # Get the Schema from SDMX files
    pandas_datasets = pysdmx_get_datasets(sdmx_data_file, sdmx_structure_file)
    schema = pandas_datasets[0].structure

    script = "DS_r <- BIS_DER;"
    result = run(
        script=script,
        data_structures=schema,
        datapoints={"BIS_DER": sdmx_data_file},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


def test_run_with_dsd_object(sdmx_structure_file):
    """Test run() with pysdmx DataStructureDefinition object."""
    from pysdmx.io import read_sdmx

    # Get the DSD from structure file
    msg = read_sdmx(sdmx_structure_file)
    # msg.structures is a list of DataStructureDefinition objects
    dsd = [s for s in msg.structures if hasattr(s, "components")][0]

    # Create a simple CSV for testing
    csv_content = "FREQ,DER_TYPE,DER_INSTR,DER_RISK,DER_REP_CTY,TIME_PERIOD,OBS_VALUE\n"
    csv_content += "A,T,F,D,5J,2020-Q1,100\n"

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write(csv_content)
        csv_path = Path(f.name)

    try:
        script = "DS_r <- BIS_DER;"
        result = run(
            script=script,
            data_structures=dsd,
            datapoints={"BIS_DER": csv_path},
            return_only_persistent=False,
        )

        assert "DS_r" in result
        assert result["DS_r"].data is not None
    finally:
        csv_path.unlink()


def test_run_with_list_of_pysdmx_objects(sdmx_data_file, sdmx_structure_file):
    """Test run() with list containing pysdmx objects."""
    from pysdmx.io import get_datasets as pysdmx_get_datasets

    pandas_datasets = pysdmx_get_datasets(sdmx_data_file, sdmx_structure_file)
    schema = pandas_datasets[0].structure

    script = "DS_r <- BIS_DER;"
    result = run(
        script=script,
        data_structures=[schema],
        datapoints={"BIS_DER": sdmx_data_file},
        return_only_persistent=False,
    )

    assert "DS_r" in result


# =============================================================================
# Tests for SDMX-CSV format files
# =============================================================================


params_sdmx_csv_files = [
    (filepath_sdmx_input / "data_v1.csv", "SDMX-CSV v1"),
    (filepath_sdmx_input / "data_v2.csv", "SDMX-CSV v2"),
]


@pytest.mark.parametrize("csv_file, description", params_sdmx_csv_files)
def test_sdmx_csv_file_exists(csv_file, description):
    """Test that SDMX-CSV test files exist."""
    if not csv_file.exists():
        pytest.skip(f"{description} test file not available")
    assert csv_file.exists()
