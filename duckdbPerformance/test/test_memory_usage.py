import pytest
from test_handlers import execute_test, OUTPUT_DIR, DATA_DIR
from duckdbPerformance.mem_gra import plot_last_memory_timeline

Params = [
    (
        [DATA_DIR / "dp" / "DS_9.csv"],
        [DATA_DIR / "ds" / "DS_9.json"],
        "DS_r <- DS_9 + 10;",
        "1GB"
    ),
    (
        [DATA_DIR / "dp" / "DS_10.csv"],
        [DATA_DIR / "ds" / "DS_10.json"],
        "DS_r <- DS_10 * 2;",
        "2GB"
    ),
]

@pytest.mark.parametrize("csv_files, ds_files, vtl_script, base_memory_limit", Params)
def test_memory_usage(csv_files, ds_files, vtl_script, base_memory_limit):
    execute_test(
        csv_paths=csv_files,
        ds_paths=ds_files,
        script=vtl_script,
        base_memory_limit=base_memory_limit,
        output_folder=OUTPUT_DIR
    )
    plot_last_memory_timeline(OUTPUT_DIR)