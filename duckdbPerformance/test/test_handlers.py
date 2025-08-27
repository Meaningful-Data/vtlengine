import csv
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

from duckdbPerformance.performance_utils.mem_gra import plot_last_memory_timeline
from duckdbPerformance.test.fusion_data import run_pipeline
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
    for item in output_folder.iterdir():
        if item.name in ("logs", ".gitignore"):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def list_output_files(output_folder: Path):
    if not output_folder.exists():
        return ["Output folder does not exist"]
    files = list(output_folder.glob("*.csv"))
    return [f"{f.name} ({f.stat().st_size / (1024**2):.2f} MB)" for f in files]


def execute_test(csv_paths, ds_paths, script, base_memory_limit, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    id_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    csv_names = csv_paths.name
    json_names = ds_paths.name

    print(
        f"\nExecuting test:\n CSVs: {csv_names}\n JSONs: {json_names}\n"
        f"Script: {script}\n Memory limit: {base_memory_limit}\n Output folder: {output_folder}",
        flush=True,
    )

    ConnectionManager.configure(memory_limit=base_memory_limit)

    stop_timer = threading.Event()
    proc = psutil.Process()

    def timer_loop():
        start = time.time()
        while not stop_timer.wait(1):
            elapsed = time.time() - start
            try:
                rss = proc.memory_info().rss / (1024**2)
                print(f"[Timer] {elapsed:.1f}s | Memory: {rss:.2f} MB", flush=True)
            except psutil.Error:
                print(f"[Timer] {elapsed:.1f}s | Memory info unavailable", flush=True)

    timer_thread = threading.Thread(target=timer_loop, daemon=True)
    timer_thread.start()

    with MemAnalyzer(interval_s=0.01, keep_series=True) as ma:
        start_time = time.time()
        result = run(  # noqa: F841
            script=script,
            data_structures=ds_paths,
            datapoints=csv_paths,
            output_folder=output_folder,
        )
        duration = time.time() - start_time

    stop_timer.set()
    timer_thread.join(timeout=1)

    mem_series_file = output_folder / f"mem_series_{id_}.csv"
    with open(mem_series_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perf", "t_rel_s", "rss_bytes", "duck_bytes"])
        w.writerows(ma.series)

    timeline_path = run_pipeline()
    timeline_file_str = timeline_path.name if timeline_path else "Not found"

    save_results(
        id_,
        file_csv=csv_names,
        file_json=json_names,
        mem_limit=base_memory_limit,
        duration_sec=duration,
        peak_rss_mb=ma.peak_rss / (1024**2),
        output_files="; ".join(list_output_files(output_folder)),
        script=script,
        memory_timeline=timeline_file_str,
    )
    if timeline_path:
        plot_last_memory_timeline(timeline_path, output_dir=output_folder)

    print("--- SUMMARY ---")
    print(
        f"Duration: {duration:.2f}s | Peak RSS: "
        f"{ma.peak_rss / (1024**2):.2f} MB | Timeline: {timeline_file_str}"
    )


def save_results(
    id_,
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
