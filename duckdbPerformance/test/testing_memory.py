import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from vtlengine import run
from vtlengine.API import create_ast
from vtlengine.AST.ASTString import ASTString
from vtlengine.connection import ConnectionManager

id_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "BIG_DATA"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = BASE_DIR / "test" / "test_results.csv"

duckdb_logs_path = OUTPUT_DIR / "logs.json"
duckdb_logs_path.parent.mkdir(parents=True, exist_ok=True)
os.environ["DUCKDB_LOGS"] = str(duckdb_logs_path)


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
        self.series = []

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
            self.series.append((perf_abs, t_rel, rss, 0))


def remove_outputs(output_folder: Path):
    if not output_folder.exists():
        return
    for file in output_folder.glob("*.csv"):
        os.remove(file)


def list_output_files(output_folder: Path):
    if not output_folder.exists():
        return ["Output folder does not exist"]
    files = list(output_folder.glob("*.csv"))
    return [f"{f.name} ({f.stat().st_size / (1024**2):.2f} MB)" for f in files]


def execute_test(csv_paths: list[Path], ds_paths: list[Path], script: str, base_memory_limit: str, output_folder: Path):
    csv_names = "; ".join(p.name for p in csv_paths)
    json_names = "; ".join(p.name for p in ds_paths)

    print(
        f"Executing test:\n CSVs: {csv_names}\n JSONs: {json_names}\n "
        f"Memory limit: {base_memory_limit}\n Output folder: {output_folder}"
    )
    ConnectionManager.configure(memory_limit=base_memory_limit)
    output_folder.mkdir(parents=True, exist_ok=True)
    remove_outputs(output_folder)

    stop_timer = threading.Event()
    proc = psutil.Process(os.getpid())

    def timer_loop():
        start = time.time()
        while not stop_timer.wait(1):
            elapsed = time.time() - start
            try:
                rss = proc.memory_info().rss / (1024**2)
                print(f"[Timer] {elapsed:.1f} s | Memory: {rss:.2f} MB")
            except psutil.Error:
                print(f"[Timer] {elapsed:.1f} s | (Unable to read memory info)")

    timer_thread = threading.Thread(target=timer_loop, daemon=True)
    timer_thread.start()

    with MemAnalyzer(pid=os.getpid(), interval_s=0.01, keep_series=True) as ma:
        start_time = time.time()
        result = run(script=script, data_structures=ds_paths, datapoints=csv_paths, output_folder=output_folder)
        duration = time.time() - start_time

    stop_timer.set()
    timer_thread.join(timeout=1)

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

    import fusion_data
    fusion_data.main()
    timeline_files = list(output_folder.glob("memory_timeline_*.csv"))
    timeline_file = max(timeline_files, key=lambda f: f.stat().st_mtime) if timeline_files else None
    timeline_file_str = timeline_file.name if timeline_file else "Not found"

    save_results(
        file_csv=csv_names,
        file_json=json_names,
        mem_limit=base_memory_limit,
        duration_sec=duration,
        peak_rss_mb=peak_rss_mb,
        output_files="; ".join(output_files),
        script=script,
        memory_timeline=timeline_file_str
    )
    print("\n--- SUMMARY ---")
    print(f"Duration: {duration:.2f} s")
    print(f"Peak RSS: {peak_rss_mb:.2f} MB")
    print(f"Output files: {output_files}")
    print(f"Run result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")


def save_results(
    file_csv,
    file_json,
    mem_limit,
    duration_sec,
    peak_rss_mb,
    output_files,
    script,
    memory_timeline=None,
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
                    "CSV Files",
                    "JSON Files",
                    "Memory Limit",
                    "Duration (s)",
                    "Peak RSS (MB)",
                    "Output Files",
                    "Memory Timeline",
                ]
            )
        script = "".join(ASTString(pretty=False).render(create_ast(script)).splitlines())
        writer.writerow(
            [
                id_,
                datetime.now().isoformat(timespec="seconds"),
                script,
                file_csv,
                file_json,
                mem_limit,
                f"{duration_sec:.2f}",
                f"{peak_rss_mb:.2f}",
                output_files,
                memory_timeline,
            ]
        )


if __name__ == "__main__":
    csv_file = [(DATA_DIR / "dp" / "DS_10.csv"), (DATA_DIR / "dp" / "DS_9.csv")]
    ds_file = [DATA_DIR / "ds" / "DS_10.json", DATA_DIR / "ds" / "DS_9.json"]
    vtl_script = "DS_r <- DS_10 > DS_9;"
    execute_test(csv_file, ds_file, vtl_script, base_memory_limit="1GB", output_folder=OUTPUT_DIR)
    __import__("fusion_data").main()