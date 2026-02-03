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
from pysdmx.model import DataflowRef, Reference, Ruleset, TransformationScheme, UserDefinedOperator
from pysdmx.model.dataflow import Dataflow, Schema
from pysdmx.model.vtl import VtlDataflowMapping

from tests.Helper import TestHelper
from vtlengine.API import generate_sdmx, prettify, run, run_sdmx, semantic_analysis
from vtlengine.API._InternalApi import _check_script, to_vtl_json
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
            # Use BIS_DER which matches the structure from sdmx_data_structure fixture
            with pytest.raises(DataLoadError, match=error_match):
                run(
                    script="DS_r <- BIS_DER;",
                    data_structures=sdmx_data_structure,
                    datapoints={"BIS_DER": test_file},
                )
        finally:
            test_file.unlink()
    elif error_type == "nonexistent":
        with pytest.raises(DataLoadError, match=error_match):
            run(
                script="DS_r <- BIS_DER;",
                data_structures=sdmx_data_structure,
                datapoints={"BIS_DER": Path(file_or_content)},
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


# =============================================================================
# Integration tests for mixed SDMX inputs
# =============================================================================


def test_run_sdmx_structure_with_sdmx_datapoints(sdmx_data_file, sdmx_structure_file):
    """Test run() with both SDMX structure and SDMX datapoints."""
    script = "DS_r <- BIS_DER;"
    result = run(
        script=script,
        data_structures=sdmx_structure_file,
        datapoints={"BIS_DER": sdmx_data_file},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


def test_run_schema_with_csv_datapoints(sdmx_data_file, sdmx_structure_file):
    """Test run() with pysdmx Schema and plain CSV datapoints."""
    from pysdmx.io import get_datasets as pysdmx_get_datasets

    pandas_datasets = pysdmx_get_datasets(sdmx_data_file, sdmx_structure_file)
    schema = pandas_datasets[0].structure

    # Create CSV with same structure
    csv_content = "FREQ,DER_TYPE,DER_INSTR,DER_RISK,DER_REP_CTY,TIME_PERIOD,OBS_VALUE\n"
    csv_content += "A,T,F,D,5J,2020-Q1,100\n"

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write(csv_content)
        csv_path = Path(f.name)

    try:
        script = "DS_r <- BIS_DER;"
        result = run(
            script=script,
            data_structures=schema,
            datapoints={"BIS_DER": csv_path},
            return_only_persistent=False,
        )

        assert "DS_r" in result
        assert result["DS_r"].data is not None
    finally:
        csv_path.unlink()


def test_run_sdmx_structure_error_invalid_file(sdmx_data_file):
    """Test error handling for invalid SDMX structure file."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
        f.write("<invalid>not sdmx structure</invalid>")
        invalid_structure = Path(f.name)

    try:
        with pytest.raises(DataLoadError, match="0-3-1-11"):
            run(
                script="DS_r <- TEST;",
                data_structures=invalid_structure,
                datapoints={"TEST": sdmx_data_file},
            )
    finally:
        invalid_structure.unlink()


# =============================================================================
# Tests for semantic_analysis() with SDMX structures
# =============================================================================


def test_semantic_analysis_with_sdmx_structure_file(sdmx_structure_file):
    """Test semantic_analysis() with SDMX structure file path."""
    script = "DS_r <- BIS_DER;"
    result = semantic_analysis(
        script=script,
        data_structures=sdmx_structure_file,
    )

    assert "DS_r" in result
    assert isinstance(result["DS_r"], Dataset)


def test_semantic_analysis_with_sdmx_structure_file_list(sdmx_structure_file):
    """Test semantic_analysis() with list of SDMX structure files."""
    script = "DS_r <- BIS_DER;"
    result = semantic_analysis(
        script=script,
        data_structures=[sdmx_structure_file],
    )

    assert "DS_r" in result
    assert isinstance(result["DS_r"], Dataset)


def test_semantic_analysis_with_schema_object(sdmx_data_file, sdmx_structure_file):
    """Test semantic_analysis() with pysdmx Schema object."""
    from pysdmx.io import get_datasets as pysdmx_get_datasets

    pandas_datasets = pysdmx_get_datasets(sdmx_data_file, sdmx_structure_file)
    schema = pandas_datasets[0].structure

    script = "DS_r <- BIS_DER;"
    result = semantic_analysis(
        script=script,
        data_structures=schema,
    )

    assert "DS_r" in result
    assert isinstance(result["DS_r"], Dataset)


def test_semantic_analysis_with_dsd_object(sdmx_structure_file):
    """Test semantic_analysis() with pysdmx DataStructureDefinition object."""
    from pysdmx.io import read_sdmx

    msg = read_sdmx(sdmx_structure_file)
    dsd = [s for s in msg.structures if hasattr(s, "components")][0]

    script = "DS_r <- BIS_DER;"
    result = semantic_analysis(
        script=script,
        data_structures=dsd,
    )

    assert "DS_r" in result
    assert isinstance(result["DS_r"], Dataset)


def test_semantic_analysis_with_dataflow_object_error():
    """Test semantic_analysis() error when Dataflow has no associated DSD."""
    # A Dataflow without associated DSD should raise an error
    dataflow = Dataflow(id="BIS_DER", agency="BIS", version="1.0")

    script = "DS_r <- BIS_DER;"
    with pytest.raises(InputValidationException, match="has no associated DataStructureDefinition"):
        semantic_analysis(
            script=script,
            data_structures=dataflow,
        )


def test_semantic_analysis_with_list_of_pysdmx_objects(sdmx_data_file, sdmx_structure_file):
    """Test semantic_analysis() with list of pysdmx objects."""
    from pysdmx.io import get_datasets as pysdmx_get_datasets

    pandas_datasets = pysdmx_get_datasets(sdmx_data_file, sdmx_structure_file)
    schema = pandas_datasets[0].structure

    script = "DS_r <- BIS_DER;"
    result = semantic_analysis(
        script=script,
        data_structures=[schema],
    )

    assert "DS_r" in result
    assert isinstance(result["DS_r"], Dataset)


def test_semantic_analysis_error_invalid_sdmx_structure():
    """Test semantic_analysis() error handling for invalid SDMX structure file."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
        f.write("<invalid>not sdmx structure</invalid>")
        invalid_structure = Path(f.name)

    try:
        with pytest.raises(DataLoadError, match="0-3-1-11"):
            semantic_analysis(
                script="DS_r <- TEST;",
                data_structures=invalid_structure,
            )
    finally:
        invalid_structure.unlink()


# =============================================================================
# Tests for run() with sdmx_mappings parameter
# =============================================================================


def test_run_with_sdmx_mappings_dict(sdmx_data_file, sdmx_structure_file):
    """Test run() with sdmx_mappings as dict."""
    script = "DS_r <- DS_1;"
    result = run(
        script=script,
        data_structures=sdmx_structure_file,
        datapoints={"DS_1": sdmx_data_file},
        sdmx_mappings={"DataStructure=BIS:BIS_DER(1.0)": "DS_1"},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


def test_run_with_sdmx_mappings_vtl_dataflow_mapping(sdmx_data_file, sdmx_structure_file):
    """Test run() with sdmx_mappings as VtlDataflowMapping object."""
    from pysdmx.io import get_datasets as pysdmx_get_datasets

    # Get the actual schema URN from the SDMX files
    pandas_datasets = pysdmx_get_datasets(sdmx_data_file, sdmx_structure_file)
    schema = pandas_datasets[0].structure

    script = "DS_r <- DS_1;"
    mapping = VtlDataflowMapping(
        dataflow=schema.short_urn,
        dataflow_alias="DS_1",
        id="VTL_MAP_1",
    )
    result = run(
        script=script,
        data_structures=schema,
        datapoints={"DS_1": sdmx_data_file},
        sdmx_mappings=mapping,
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


def test_run_with_sdmx_mappings_and_schema_object(sdmx_data_file, sdmx_structure_file):
    """Test run() with Schema object and sdmx_mappings."""
    from pysdmx.io import get_datasets as pysdmx_get_datasets

    pandas_datasets = pysdmx_get_datasets(sdmx_data_file, sdmx_structure_file)
    schema = pandas_datasets[0].structure

    script = "DS_r <- CUSTOM_NAME;"
    result = run(
        script=script,
        data_structures=schema,
        datapoints={"CUSTOM_NAME": sdmx_data_file},
        sdmx_mappings={schema.short_urn: "CUSTOM_NAME"},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


# =============================================================================
# Tests for run() with additional datapoints variations
# =============================================================================


def test_run_with_sdmx_datapoints_directory(sdmx_data_file, sdmx_data_structure):
    """Test run() with directory containing SDMX files as datapoints."""
    # Create a temp directory with only the data file
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        # Copy only the data file to the temp directory
        dest_file = Path(tmpdir) / sdmx_data_file.name
        shutil.copy(sdmx_data_file, dest_file)

        script = "DS_r <- BIS_DER;"
        result = run(
            script=script,
            data_structures=sdmx_data_structure,
            datapoints=Path(tmpdir),
            return_only_persistent=False,
        )

        assert "DS_r" in result
        assert result["DS_r"].data is not None


def test_run_with_sdmx_datapoints_list_paths(sdmx_data_file, sdmx_data_structure):
    """Test run() with list of SDMX file paths as datapoints."""
    script = "DS_r <- BIS_DER;"
    result = run(
        script=script,
        data_structures=sdmx_data_structure,
        datapoints=[sdmx_data_file],
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


def test_run_with_sdmx_datapoints_dataframe(sdmx_data_file, sdmx_structure_file):
    """Test run() with DataFrame from SDMX file as datapoints."""
    from pysdmx.io import get_datasets as pysdmx_get_datasets

    pandas_datasets = pysdmx_get_datasets(sdmx_data_file, sdmx_structure_file)
    schema = pandas_datasets[0].structure
    df = pandas_datasets[0].data

    script = "DS_r <- BIS_DER;"
    result = run(
        script=script,
        data_structures=schema,
        datapoints={"BIS_DER": df},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


# =============================================================================
# Tests for run_sdmx() with additional mapping types
# =============================================================================


def test_run_sdmx_with_dataflow_object_mapping():
    """Test run_sdmx() with Dataflow object in VtlDataflowMapping."""
    data_file = filepath_sdmx_input / "str_all_minimal_df.xml"
    structure_file = filepath_sdmx_input / "metadata_minimal_df.xml"

    datasets = get_datasets(data_file, structure_file)
    mapping = VtlDataflowMapping(
        dataflow=Dataflow(id="TEST_DF", agency="MD", version="1.0"),
        dataflow_alias="DS_1",
        id="VTL_MAP_DF",
    )

    script = "DS_r := DS_1 [calc Me_4 := OBS_VALUE];"
    result = run_sdmx(script, datasets, mappings=mapping, return_only_persistent=False)

    assert "DS_r" in result
    assert isinstance(result["DS_r"].data, pd.DataFrame)


def test_run_sdmx_with_reference_mapping():
    """Test run_sdmx() with Reference object in VtlDataflowMapping."""
    data_file = filepath_sdmx_input / "str_all_minimal_df.xml"
    structure_file = filepath_sdmx_input / "metadata_minimal_df.xml"

    datasets = get_datasets(data_file, structure_file)
    mapping = VtlDataflowMapping(
        dataflow=Reference(sdmx_type="Dataflow", agency="MD", id="TEST_DF", version="1.0"),
        dataflow_alias="DS_1",
        id="VTL_MAP_REF",
    )

    script = "DS_r := DS_1 [calc Me_4 := OBS_VALUE];"
    result = run_sdmx(script, datasets, mappings=mapping, return_only_persistent=False)

    assert "DS_r" in result
    assert isinstance(result["DS_r"].data, pd.DataFrame)


def test_run_sdmx_with_dataflow_ref_mapping():
    """Test run_sdmx() with DataflowRef object in VtlDataflowMapping."""
    data_file = filepath_sdmx_input / "str_all_minimal_df.xml"
    structure_file = filepath_sdmx_input / "metadata_minimal_df.xml"

    datasets = get_datasets(data_file, structure_file)
    mapping = VtlDataflowMapping(
        dataflow=DataflowRef(agency="MD", id="TEST_DF", version="1.0"),
        dataflow_alias="DS_1",
        id="VTL_MAP_DFREF",
    )

    script = "DS_r := DS_1 [calc Me_4 := OBS_VALUE];"
    result = run_sdmx(script, datasets, mappings=mapping, return_only_persistent=False)

    assert "DS_r" in result
    assert isinstance(result["DS_r"].data, pd.DataFrame)


# =============================================================================
# Tests for run_sdmx() error cases with mappings
# =============================================================================


def test_run_sdmx_error_missing_mapping_for_multiple_datasets():
    """Test run_sdmx() error when multiple datasets but no mapping provided."""
    datasets = [
        PandasDataset(
            structure=Schema(id="DS1", components=[], agency="BIS", context="datastructure"),
            data=pd.DataFrame(),
        ),
        PandasDataset(
            structure=Schema(id="DS2", components=[], agency="BIS", context="datastructure"),
            data=pd.DataFrame(),
        ),
    ]
    with pytest.raises(InputValidationException, match="0-1-3-3"):
        run_sdmx("DS_r := DS1;", datasets)


def test_run_sdmx_error_invalid_mapping_type():
    """Test run_sdmx() error when invalid mapping type provided."""
    datasets = [
        PandasDataset(
            structure=Schema(id="BIS_DER", components=[], agency="BIS", context="datastructure"),
            data=pd.DataFrame(),
        )
    ]
    with pytest.raises(InputValidationException, match="Expected dict or VtlDataflowMapping"):
        run_sdmx("DS_r := BIS_DER;", datasets, mappings="invalid_type")


def test_run_sdmx_error_invalid_dataflow_type_in_mapping():
    """Test run_sdmx() error when invalid dataflow type in VtlDataflowMapping."""
    datasets = [
        PandasDataset(
            structure=Schema(id="BIS_DER", components=[], agency="BIS", context="datastructure"),
            data=pd.DataFrame(),
        )
    ]
    mapping = VtlDataflowMapping(dataflow=123, dataflow_alias="ALIAS", id="Test")
    with pytest.raises(
        InputValidationException,
        match="Expected str, Reference, DataflowRef or Dataflow type for dataflow",
    ):
        run_sdmx("DS_r := BIS_DER;", datasets, mappings=mapping)


def test_run_sdmx_error_dataset_not_in_script():
    """Test run_sdmx() error when mapped dataset name not found in script."""
    data_file = filepath_sdmx_input / "str_all_minimal_df.xml"
    structure_file = filepath_sdmx_input / "metadata_minimal_df.xml"

    datasets = get_datasets(data_file, structure_file)
    mapping = {"Dataflow=MD:TEST_DF(1.0)": "NONEXISTENT_NAME"}

    with pytest.raises(InputValidationException, match="0-1-3-5"):
        run_sdmx("DS_r := DS_1;", datasets, mappings=mapping)


def test_run_sdmx_error_invalid_datasets_type():
    """Test run_sdmx() error when datasets is not a list of PandasDataset."""
    with pytest.raises(InputValidationException, match="0-1-3-7"):
        run_sdmx("DS_r := TEST;", "not_a_list")


def test_run_sdmx_error_schema_not_in_mapping():
    """Test run_sdmx() error when schema URN not found in mapping."""
    datasets = [
        PandasDataset(
            structure=Schema(id="OTHER_DS", components=[], agency="BIS", context="datastructure"),
            data=pd.DataFrame(),
        )
    ]
    mapping = {"Dataflow=MD:DIFFERENT(1.0)": "DS_1"}

    with pytest.raises(InputValidationException, match="0-1-3-4"):
        run_sdmx("DS_r := DS_1;", datasets, mappings=mapping)


# =============================================================================
# Tests for semantic_analysis() error cases
# =============================================================================


def test_semantic_analysis_error_nonexistent_sdmx_file():
    """Test semantic_analysis() error for nonexistent SDMX structure file."""
    with pytest.raises(DataLoadError, match="0-3-1-1"):
        semantic_analysis(
            script="DS_r <- TEST;",
            data_structures=Path("/nonexistent/structure.xml"),
        )


# =============================================================================
# Tests for run() error cases with SDMX inputs
# =============================================================================


def test_run_error_nonexistent_sdmx_datapoint():
    """Test run() error for nonexistent SDMX datapoint file."""
    structure_file = filepath_json / "DS_1.json"
    with open(structure_file) as f:
        data_structure = json.load(f)

    with pytest.raises(DataLoadError, match="0-3-1-1"):
        run(
            script="DS_r <- DS_1;",
            data_structures=data_structure,
            datapoints={"DS_1": Path("/nonexistent/data.xml")},
        )


def test_run_error_invalid_sdmx_datapoint():
    """Test run() error for invalid SDMX datapoint file."""
    structure_file = filepath_json / "DS_1.json"
    with open(structure_file) as f:
        data_structure = json.load(f)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
        f.write("<invalid>not sdmx data</invalid>")
        invalid_data = Path(f.name)

    try:
        with pytest.raises(DataLoadError, match="0-3-1-8"):
            run(
                script="DS_r <- DS_1;",
                data_structures=data_structure,
                datapoints={"DS_1": invalid_data},
            )
    finally:
        invalid_data.unlink()


# =============================================================================
# Tests for combined SDMX structures and datapoints with mappings
# =============================================================================


def test_run_full_sdmx_workflow_with_mappings(sdmx_data_file, sdmx_structure_file):
    """Test complete SDMX workflow with structure file, datapoints, and mappings."""
    script = "DS_r <- CUSTOM_DS;"

    result = run(
        script=script,
        data_structures=sdmx_structure_file,
        datapoints={"CUSTOM_DS": sdmx_data_file},
        sdmx_mappings={"DataStructure=BIS:BIS_DER(1.0)": "CUSTOM_DS"},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None
    assert len(result["DS_r"].data) > 0


def test_run_with_dsd_and_sdmx_mappings(sdmx_data_file, sdmx_structure_file):
    """Test run() with DSD object and sdmx_mappings."""
    from pysdmx.io import read_sdmx

    msg = read_sdmx(sdmx_structure_file)
    dsd = [s for s in msg.structures if hasattr(s, "components")][0]

    script = "DS_r <- MAPPED_NAME;"
    result = run(
        script=script,
        data_structures=dsd,
        datapoints={"MAPPED_NAME": sdmx_data_file},
        sdmx_mappings={dsd.short_urn: "MAPPED_NAME"},
        return_only_persistent=False,
    )

    assert "DS_r" in result
    assert result["DS_r"].data is not None


# =============================================================================
# Tests for generate_sdmx() function
# =============================================================================


def test_generate_sdmx_without_udo_or_rs():
    """Test generate_sdmx() with simple transformation (no UDO or Ruleset)."""
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


def test_generate_sdmx_with_udo():
    """Test generate_sdmx() with User Defined Operator."""
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


def test_generate_sdmx_with_dp_ruleset():
    """Test generate_sdmx() with datapoint ruleset."""
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


def test_generate_sdmx_with_hierarchical_ruleset():
    """Test generate_sdmx() with hierarchical ruleset."""
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


def test_generate_sdmx_with_2_rulesets():
    """Test generate_sdmx() with multiple rulesets."""
    script = base_path / "data" / "vtl" / "validations.vtl"
    ts = generate_sdmx(script, agency_id="MD", id="TestID")
    assert isinstance(ts, TransformationScheme)
    rs_scheme = ts.ruleset_schemes[0]
    assert rs_scheme.id == "RS1"
    assert len(rs_scheme.items) == 2
    assert isinstance(rs_scheme.items[0], Ruleset)
    assert rs_scheme.items[0].ruleset_type == "datapoint"


def test_generate_sdmx_with_ruleset_and_udo():
    """Test generate_sdmx() with both ruleset and UDO."""
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


def test_generate_sdmx_and_check_script():
    """Test generate_sdmx() and verify script can be regenerated."""
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
    """Test generate_sdmx() with valuedomain ruleset and verify script regeneration."""
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


# =============================================================================
# Tests for Memory-Efficient Pattern with SDMX Files (Issue #470)
# =============================================================================


def test_sdmx_memory_efficient_with_output_folder(sdmx_data_file, sdmx_data_structure):
    """
    Test that SDMX-ML files work with memory-efficient pattern (output_folder).

    When output_folder is provided:
    1. SDMX-ML file paths are stored for lazy loading (not loaded upfront)
    2. Data is loaded on-demand during execution via load_datapoints
    3. Results are written to disk

    This test verifies Issue #470 - QA for SDMX memory-efficient loading.
    """
    script = "DS_r <- BIS_DER;"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(
            script=script,
            data_structures=sdmx_data_structure,
            datapoints={"BIS_DER": sdmx_data_file},
            output_folder=tmpdir,
            return_only_persistent=False,
        )

        # Result should contain DS_r
        assert "DS_r" in result
        assert isinstance(result["DS_r"], Dataset)

        # Output file should exist and have correct content
        output_file = Path(tmpdir) / "DS_r.csv"
        assert output_file.exists(), "Output file DS_r.csv should be created"
        df = pd.read_csv(output_file)
        assert len(df) == 10, "Should have 10 rows from SDMX data"


def test_sdmx_memory_efficient_with_persistent_assignment(sdmx_data_file, sdmx_data_structure):
    """
    Test SDMX-ML with persistent assignment and output_folder.

    Persistent assignments (using <-) should have their results saved to disk.
    """
    script = "DS_r <- BIS_DER [calc Me_4 := OBS_VALUE];"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(
            script=script,
            data_structures=sdmx_data_structure,
            datapoints={"BIS_DER": sdmx_data_file},
            output_folder=tmpdir,
            return_only_persistent=True,
        )

        # Should only return persistent dataset
        assert "DS_r" in result

        # Verify output file exists and has content
        output_file = Path(tmpdir) / "DS_r.csv"
        assert output_file.exists(), "Output file DS_r.csv should exist"
        df = pd.read_csv(output_file)
        assert len(df) == 10, "Should have 10 rows"
        assert "Me_4" in df.columns, "Should have calculated measure Me_4"


def test_sdmx_memory_efficient_multi_step_transformation(sdmx_data_file, sdmx_data_structure):
    """
    Test SDMX-ML with multi-step transformation and memory-efficient pattern.

    This tests that intermediate results are properly managed and SDMX-ML data
    is loaded via load_datapoints during execution.
    """
    # Use a filter on FREQ which is a String identifier
    script = """
    DS_temp := BIS_DER [filter FREQ = "A"];
    DS_r <- DS_temp [calc Me_4 := OBS_VALUE || "_transformed"];
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(
            script=script,
            data_structures=sdmx_data_structure,
            datapoints={"BIS_DER": sdmx_data_file},
            output_folder=tmpdir,
            return_only_persistent=True,
        )

        # Only persistent assignment should be in result
        assert "DS_r" in result
        assert "DS_temp" not in result  # Non-persistent, not returned

        # Output file should exist with transformed data
        output_file = Path(tmpdir) / "DS_r.csv"
        assert output_file.exists()
        df = pd.read_csv(output_file)
        assert len(df) > 0, "Should have data after transformation"
        assert "Me_4" in df.columns, "Should have calculated measure Me_4"


def test_mixed_sdmx_csv_memory_efficient(sdmx_data_file, sdmx_data_structure):
    """
    Test memory-efficient pattern with mixed SDMX-ML and plain CSV files.

    Both SDMX-ML and plain CSV files should be loaded on-demand during execution
    via load_datapoints which supports both formats.
    """
    # Get CSV structure
    csv_structure_path = filepath_json / "DS_1.json"
    with open(csv_structure_path) as f:
        csv_structure = json.load(f)

    # Combine structures
    combined_structure = {"datasets": sdmx_data_structure["datasets"] + csv_structure["datasets"]}

    script = "DS_r <- BIS_DER; DS_r2 <- DS_1;"
    csv_file = filepath_csv / "DS_1.csv"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(
            script=script,
            data_structures=combined_structure,
            datapoints={
                "BIS_DER": sdmx_data_file,
                "DS_1": csv_file,
            },
            output_folder=tmpdir,
            return_only_persistent=False,
        )

        # Both results should be present
        assert "DS_r" in result
        assert "DS_r2" in result

        # Both output files should exist
        assert (Path(tmpdir) / "DS_r.csv").exists()
        assert (Path(tmpdir) / "DS_r2.csv").exists()
