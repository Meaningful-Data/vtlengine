import os
import time
import threading
import psutil

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
            duck_bytes = 0
            self.series.append((perf_abs, t_rel, rss, duck_bytes))
