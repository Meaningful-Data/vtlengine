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
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        perf_now = time.perf_counter()
        t_rel = perf_now - peaks_log["perf0"]
        peaks_log["series"].append((perf_now, t_rel, mem_rss, mem_duck))

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


class MemAnalyzer:
    def __init__(self, pid=None, interval_s: float = 0.1, keep_series: bool = True):
        self.pid = pid or os.getpid()
        self.interval_s = interval_s
        self.keep_series = keep_series
        self.proc = psutil.Process(self.pid)
        self._stop = threading.Event()
        self._thr = None
        self.t0 = 0.0
        self.t1 = 0.0
        self.peak_rss = 0
        self.rss_start = None
        self.rss_end = None
        self.series = []  # (perf_abs, t_rel_s, rss_bytes, duck_bytes)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def start(self):
        if self._thr is not None:
            return
        self._stop.clear()
        self.t0 = time.perf_counter()
        self._thr = threading.Thread(target=self._loop, name="PsutilMemSampler", daemon=True)
        self._thr.start()

    def stop(self):
        if self._thr is None:
            return
        self._stop.set()
        self._thr.join()
        self.t1 = time.perf_counter()

    def _loop(self):
        self._sample_once()
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self):
        try:
            rss = self.proc.memory_info().rss
        except psutil.Error:
            self._stop.set()
            return
        if self.rss_start is None:
            self.rss_start = rss
        self.rss_end = rss
        if rss > self.peak_rss:
            self.peak_rss = rss
        if self.keep_series:
            perf_abs = time.perf_counter()
            t_rel = perf_abs - self.t0
            duck = 0  # si queréis, aquí podéis integrar duckdb_memory()
            self.series.append((perf_abs, t_rel, rss, duck))


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
    output_folder.mkdir(parents=True, exist_ok=True)
    remove_outputs(output_folder)

    with MemAnalyzer(pid=os.getpid(), interval_s=0.01, keep_series=True) as ma:
        start_time = time.time()
        result = run(
            script=script, data_structures=ds_path, datapoints=csv_path, output_folder=output_folder
        )
        duration = time.time() - start_time

    # picos con MemAnalyzer
    peak_rss_mb = ma.peak_rss / (1024**2)
    # si se quiere timestamp del pico:
    if ma.series:
        peak_idx = max(range(len(ma.series)), key=lambda i: ma.series[i][2])
        peak_rel = ma.series[peak_idx][1]
    else:
        peak_idx = 0
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
    output_file = OUTPUT_DIR

    if os.path.exists(output_file):
        print("check data_o")

    if os.path.exists(csv_file):
        print("check data_p")

    if os.path.exists(ds_file):
        print("check data_s")

    vtl_script = """
        DS_r <- DS_2[calc result:= Me_1 * 10];
    """
    execute_test(csv_file, ds_file, vtl_script, base_memory_limit="4GB", output_folder=OUTPUT_DIR)