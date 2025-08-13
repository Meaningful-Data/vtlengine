import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

id_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
duckdb_logs_path = Path(__file__).parent.parent / "logs" / f"logs_{id_}.json"
if not duckdb_logs_path.parent.exists():
    duckdb_logs_path.parent.mkdir(parents=True, exist_ok=True)

os.environ["DUCKDB_LOGS"] = str(duckdb_logs_path)
from vtlengine import run  # noqa: E402
from vtlengine.API import create_ast  # noqa: E402
from vtlengine.AST.ASTString import ASTString  # noqa: E402
from vtlengine.connection import ConnectionManager  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "BIG_DATA"
OUTPUT_DIR = BASE_DIR / "data" / "output"
RESULTS_FILE = BASE_DIR / "test" / "test_results.csv"


def monitor_memory(
    pid, stop_event, peak_rss_holder, peak_duck_holder, peaks_log, conn, check_interval=1.0
):
    mem_duck = 0
    process = psutil.Process(pid)
    peak_rss = 0

    while not stop_event.is_set():
        try:
            mem_rss = process.memory_info().rss
        except Exception:
            mem_rss = 0

        if mem_rss > peak_rss:
            peak_rss = mem_rss

        # try:
        #     mem_df = conn.execute(
        #         "SELECT SUM(CAST(memory_usage_bytes AS BIGINT))
        #         AS total_bytes FROM duckdb_memory()"
        #     ).fetchdf()
        #     mem_duck = int(mem_df["total_bytes"].iloc[0] or 0)
        # except Exception as e:
        #     print(f"Error fetching DuckDB memory usage: {e}")
        #     mem_duck = 0
        # TODO: Add here monitoring every interval
        #   Ensure the Duckdb is also monitored here without using the connection
        if mem_rss > peak_rss_holder[0] or mem_duck > peak_duck_holder[0]:
            timestamp = time.time() - peaks_log["start_time"]
            if mem_rss > peak_rss_holder[0]:
                peak_rss_holder[0] = mem_rss
            if mem_duck > peak_duck_holder[0]:
                peak_duck_holder[0] = mem_duck
            peaks_log["records"].append((timestamp, mem_rss / (1024**2), mem_duck / (1024**2)))
            print(
                f"[{timestamp:.2f}s] New peak -> RSS: {mem_rss / (1024**2):.2f} MB | "
                f"DuckDB: {mem_duck / (1024**2):.2f} MB"
            )

        time.sleep(check_interval)


def remove_outputs(output_folder: Path):
    if not output_folder.exists():
        return

    # Remove only csv files
    for file in output_folder.glob("*.csv"):
        os.remove(file)

def list_output_files(output_folder: Path):
    if not output_folder.exists():
        return ["Output folder does not exist"]
    files = list(output_folder.glob("*.csv"))
    return [f"{f.name} ({f.stat().st_size / (1024**2):.2f} MB)" for f in files]


def execute_test(
    csv_path: Path, ds_path: Path, script: str, base_memory_limit: str, output_folder: Path
):
    # con = ConnectionManager.get_connection()
    print(
        f"Executing test:\n CSV: {csv_path}\n JSON: {ds_path}\n "
        f"Memory limit: {base_memory_limit}\n Output folder: {output_folder}"
    )

    ConnectionManager.configure(memory_limit=base_memory_limit)
    remove_outputs(output_folder)
    peak_rss_holder = [0]
    peak_duck_holder = [0]
    peaks_log = {"records": [], "start_time": time.time()}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(os.getpid(), stop_event, peak_rss_holder, peak_duck_holder, peaks_log, None, 1.0),
        daemon=True,
    )
    monitor_thread.start()

    start_time = time.time()
    result = run(
        script=script, data_structures=ds_path, datapoints=csv_path, output_folder=output_folder
    )
    duration = time.time() - start_time

    stop_event.set()
    monitor_thread.join(timeout=5.0)

    output_files = list_output_files(output_folder)

    # snapshot_file = output_folder /
    # f"usage_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # with open(snapshot_file, "w") as f:
    #     subprocess.run(["top", "-b", "-n", "1"], stdout=f)  # noqa: S603,S607

    save_results(
        file_csv=csv_path.name,
        file_json=ds_path.name,
        mem_limit=base_memory_limit,
        duration_sec=duration,
        peak_rss_mb=peak_rss_holder[0] / (1024**2),
        peak_duck_mb=peak_duck_holder[0] / (1024**2),
        output_files="; ".join(output_files),
        peaks_list=peaks_log["records"],
        script=script,
    )

    print("\n--- SUMMARY ---")
    print(f"Duration: {duration:.2f} s")
    print(f"Peak RSS: {peak_rss_holder[0] / (1024**2):.2f} MB")
    print(f"Peak DuckDB: {peak_duck_holder[0] / (1024**2):.2f} MB")
    print(f"Output files: {output_files}")
    print(f"Run result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    # print(f"Top snapshot saved to: {snapshot_file}")


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
                    "JSON",
                    "Memory Limit",
                    "Duration (s)",
                    "Peak RSS (MB)",
                    "Peak DuckDB (MB)",
                    "Output Files",
                    "Memory Peaks (t_s:RSS/DUCK_MB)",
                ]
            )
        peaks_str = "; ".join([f"{t:.2f}s:{rss:.2f}/{duck:.2f}" for t, rss, duck in peaks_list])

        script = "".join(ASTString(pretty=False).render(create_ast(script)).splitlines())
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
                peaks_str,
            ]
        )


if __name__ == "__main__":
    csv_file = DATA_DIR / "dp" / "DS_2.csv"
    ds_file = DATA_DIR / "ds" / "DS_2.json"
    vtl_script = """
    DS_r <- DS_2;
    """
    execute_test(csv_file, ds_file, vtl_script, base_memory_limit="4GB", output_folder=OUTPUT_DIR)
