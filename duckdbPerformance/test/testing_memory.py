import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

from duckdbPerformance.MemoryAnalyzer.MemAnalizer import MemAnalyzer
from vtlengine import run
from vtlengine.API import create_ast
from vtlengine.AST.ASTString import ASTString
from vtlengine.connection import ConnectionManager

# --- Paths & Config ---
id_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
duckdb_logs_path = Path(__file__).parent.parent / "logs" / f"logs_{id_}.json"
if not duckdb_logs_path.parent.exists():
    duckdb_logs_path.parent.mkdir(parents=True, exist_ok=True)
os.environ["DUCKDB_LOGS"] = str(duckdb_logs_path)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "BIG_DATA"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = BASE_DIR / "test" / "test_results.csv"

# --- Helpers ---
def remove_outputs(output_folder: Path):
    if output_folder.exists():
        for file in output_folder.glob("*.csv"):
            os.remove(file)


def list_output_files(output_folder: Path):
    if not output_folder.exists():
        return ["Output folder does not exist"]
    files = list(output_folder.glob("*.csv"))
    return [f"{f.name} ({f.stat().st_size / (1024**2):.2f} MB)" for f in files]

# --- Main Test Runner ---
def execute_test(csv_path: Path, ds_path: Path, script: str, base_memory_limit: str, output_folder: Path):
    print(
        f"Executing test:\n CSV: {csv_path}\n JSON: {ds_path}\n "
        f"Memory limit: {base_memory_limit}\n Output folder: {output_folder}"
    )

    ConnectionManager.configure(memory_limit=base_memory_limit)
    output_folder.mkdir(parents=True, exist_ok=True)
    remove_outputs(output_folder)

    with MemAnalyzer(pid=os.getpid(), interval_s=0.01, keep_series=True) as ma:
        start_time = time.time()
        result = run(
            script=script, data_structures=ds_path, datapoints=csv_path, output_folder=output_folder
        )
        duration = time.time() - start_time

    # Peak memory usage
    peak_rss_mb = ma.peak_rss / (1024**2)
    if ma.series:
        peak_idx = max(range(len(ma.series)), key=lambda i: ma.series[i][2])
        peak_rel = ma.series[peak_idx][1]
    else:
        peak_rel = 0.0

    mem_series_path = output_folder / f"mem_series_{id_}.csv"
    with open(mem_series_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perf", "t_rel_s", "rss_bytes", "duck_bytes"])
        w.writerows(ma.series)

    output_files = list_output_files(output_folder)

    save_results(
        file_csv=csv_path.name,
        file_json=ds_path.name,
        mem_limit=base_memory_limit,
        duration_sec=duration,
        peak_rss_mb=peak_rss_mb,
        peak_duck_mb=0.0,
        output_files="; ".join(output_files),
        peaks_list=[(peak_rel, peak_rss_mb, 0.0)],
        script=script,
    )

    print("\n--- SUMMARY ---")
    print(f"Duration: {duration:.2f} s")
    print(f"Peak RSS: {peak_rss_mb:.2f} MB")
    print(f"Peak DuckDB: {0.0:.2f} MB")
    print(f"Output files: {output_files}")
    print(f"Run result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")


# --- Save results to CSV ---
def save_results(
    file_csv,
    file_json,
    mem_limit,
    duration_sec,
    peak_rss_mb,
    peak_duck_mb,
    output_files,
    peaks_list,
    script,
):
    file_exists = RESULTS_FILE.exists()
    with open(RESULTS_FILE, mode="a+", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "ID",
                    "Date",
                    "VTL Script",
                    "CSV File",
                    "JSON",
                    "Memory Limit",
                    "Duration (s)",
                    "Peak RSS (MB)",
                    "Peak DuckDB (MB)",
                    "Output Files",
                    "Memory Detail CSV",
                ]
            )
        script = "".join(ASTString(pretty=False).render(create_ast(script)).splitlines())
        detail_file = RESULTS_FILE.parent / f"{id_}_memory_detail.csv"
        writer.writerow(
            [
                id_,
                datetime.now().isoformat(timespec="seconds"),
                script,
                Path(file_csv).name,
                Path(file_json).name,
                mem_limit,
                f"{duration_sec:.2f}",
                f"{peak_rss_mb:.2f}",
                f"{peak_duck_mb:.2f}",
                output_files,
                detail_file.name
            ]
        )

    with open(detail_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "RSS (MB)", "DuckDB (MB)"])
        for t, rss, duck in peaks_list:
            writer.writerow([f"{t:.2f}", f"{rss:.2f}", f"{duck:.2f}"])

if __name__ == "__main__":
    ds_name = "DS_2"
    csv_file = DATA_DIR / "dp" / f"{ds_name}.csv"
    ds_file = DATA_DIR / "ds" / f"{ds_name}.json"
    vtl_script = """
    DS_r <- DS_2;
    """
    execute_test(csv_file, ds_file, vtl_script, base_memory_limit="4GB", output_folder=OUTPUT_DIR)
    try:
        __import__("fusion_data").main()
    except ModuleNotFoundError:
        print("fusion_data module not found, skipping...")
    print(duckdb_logs_path)