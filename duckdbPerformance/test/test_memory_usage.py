import pytest
from test_handlers import DATA_DIR, OUTPUT_DIR, execute_test, remove_outputs

Params = [
    (
        [[DATA_DIR / "dp" / "DS_9.csv"], [DATA_DIR / "dp" / "DS_10.csv"]],
        [[DATA_DIR / "ds" / "DS_9.json"], [DATA_DIR / "ds" / "DS_10.json"]],
        ["DS_r <- DS_9 + 10;", "DS_r <- DS_10 * 2;"],
        "2GB",  # base_memory_limit must be set also in ConnectionManager in connection.py.
        # At the moment it is not possible to pass it directly.
    ),
]


@pytest.fixture(scope="session", autouse=True)
def clean_outputs_once():
    remove_outputs(OUTPUT_DIR)


@pytest.mark.parametrize(("csv_files", "ds_files", "vtl_script", "base_memory_limit"), Params)
def test_memory_usage(csv_files, ds_files, vtl_script, base_memory_limit):
    execute_test(
        csv_paths_list=csv_files,
        ds_paths_list=ds_files,
        scripts_list=vtl_script,
        base_memory_limit=base_memory_limit,
        output_folder=OUTPUT_DIR,
    )
