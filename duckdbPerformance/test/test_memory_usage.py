import os
import shutil
from pathlib import Path

import pytest

DATAPOINTS_DIR = Path(__file__).parent.parent / "data" / "BIG_DATA" / "dp"
DATASTRUCTURES_DIR = Path(__file__).parent.parent / "data" / "BIG_DATA" / "ds"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

Params = [
    (DATAPOINTS_DIR / "DS_9.csv", DATASTRUCTURES_DIR / "DS_9.json", "DS_r <- DS_9 + 10;", "1GB"),
    (
        DATAPOINTS_DIR / "DS_10.csv",
        DATASTRUCTURES_DIR / "DS_10.json",
        "DS_r <- DS_10 * 2;",
        "1GB",
    ),
]


def remove_outputs(output_folder: Path):
    for item in output_folder.iterdir():
        if item.name in ("logs", ".gitignore"):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


@pytest.fixture(scope="session", autouse=True)
def clean_outputs_once():
    remove_outputs(OUTPUT_DIR)


@pytest.mark.parametrize(("csv_files", "ds_files", "vtl_script", "base_memory_limit"), Params)
def test_memory_usage(csv_files, ds_files, vtl_script, base_memory_limit):
    os.environ["DUCKDB_MEMORY_LIMIT"] = base_memory_limit
    from test_handlers import execute_test

    execute_test(
        csv_paths=csv_files,
        ds_paths=ds_files,
        script=vtl_script,
        base_memory_limit=base_memory_limit,
        output_folder=OUTPUT_DIR,
    )


def main():
    for param_list in Params:
        csv_files, ds_files, vtl_script, base_memory_limit = param_list
        os.environ["DUCKDB_MEMORY_LIMIT"] = base_memory_limit
        from test_handlers import execute_test

        execute_test(
            csv_paths=csv_files,
            ds_paths=ds_files,
            script=vtl_script,
            base_memory_limit=base_memory_limit,
            output_folder=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
