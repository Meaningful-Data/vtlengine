import csv
import json
import re
from pathlib import Path

BYTES_IN_MB = 1024 ** 2
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "output"

def latest(pattern):
    files = list(OUT_DIR.glob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def load_json(p):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_mem_series(p):
    rows = []
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "perf": float(row["perf"]),
                "t_rel_s": float(row["t_rel_s"]),
                "rss": int(row["rss_bytes"]),
            })
    rows.sort(key=lambda x: x["perf"])
    return rows

def seconds(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def raw_dur(node):
    t = seconds(node.get("operator_timing", 0.0))
    if t > 0.0:
        return t
    t = seconds(node.get("cpu_time", 0.0))
    if t > 0.0:
        return t
    ch = node.get("children", []) or []
    if not ch:
        return 0.0
    return sum(raw_dur(c) for c in ch)

def flatten_windows(node, end_perf, path=None, depth=0, target_dur=None):
    path = path or []
    name = node.get("operator_name", node.get("query_name", "ROOT"))
    typ  = node.get("operator_type", "ROOT")

    dur_base = raw_dur(node)
    dur = target_dur if target_dur is not None else (seconds(node.get("latency", 0.0)) or dur_base)
    start = end_perf - dur

    rows = [{
        "path": " / ".join(path + [name]),
        "name": name,
        "type": typ,
        "start": start,
        "end": end_perf,
        "dur": dur,
        "depth": depth,
    }]

    children = node.get("children", []) or []
    if children:
        child_raw = [raw_dur(c) for c in children]
        sum_raw = sum(child_raw)

        if sum_raw > 0.0:
            child_scaled = [d * (dur / sum_raw) for d in child_raw]
        else:
            child_scaled = [dur / len(children)] * len(children)

        cur_end = end_perf
        for ch, ch_dur in zip(reversed(children), reversed(child_scaled)):
            rows += flatten_windows(ch, cur_end, path + [name], depth + 1, target_dur=ch_dur)
            cur_end -= ch_dur

    return rows

def active_op_at(perf, windows):
    cands = [w for w in windows if w["start"] <= perf <= w["end"]]
    return max(cands, key=lambda w: (w["depth"], w["dur"])) if cands else None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mem_csv = latest("mem_series_*.csv")
    if not mem_csv:
        print("No mem_series_*.csv found in output/")
        return 1

    m = re.match(r"mem_series_(.+)\.csv$", mem_csv.name)
    run_id = m.group(1) if m else "latest"

    finish = OUT_DIR / "logs" / "finish.json"
    if not finish.exists():
        print("output/logs/finish.json not found")
        return 1

    profile = OUT_DIR / "logs" / "logs.json"
    if not profile.exists():
        print("output/logs/logs.json not found")
        return 1

    series = read_mem_series(mem_csv)
    prof   = load_json(profile)
    fin    = load_json(finish)

    if "perf_end" not in fin:
        print("finish.json is missing 'perf_end'")
        return 1

    perf_end = float(fin["perf_end"])
    latency  = seconds(prof.get("latency", 0.0)) or raw_dur({"children": prof.get("children", [])})
    if latency <= 0.0:
        print("Could not determine total DuckDB latency")
        return 1

    root = {
        "operator_name": prof.get("query_name", "DUCKDB_QUERY"),
        "operator_type": "ROOT",
        "operator_timing": latency,
        "children": prof.get("children", []),
        "latency": latency,
    }

    windows = flatten_windows(root, perf_end, target_dur=latency)

    timeline_path = OUT_DIR / f"memory_timeline_{run_id}.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["perf_s","perf_ms","t_rel_s","t_rel_ms","rss_mb","is_duckdb","op_name","op_type","op_path","op_depth"])
        perf_start = perf_end - latency
        for m in series:
            op = active_op_at(m["perf"], windows)
            if op:
                is_duckdb = 1
                op_name = "DUCKDB_QUERY" if op["type"] == "ROOT" else op["name"]
                op_type = op["type"]; op_path = op["path"]; op_depth = op["depth"]
            else:
                is_duckdb = 0
                op_name = op_type = op_path = ""; op_depth = ""

            t_rel = m["perf"] - perf_start
            w.writerow([
                f"{m['perf']:.6f}",
                int(round(m["perf"] * 1000)),
                f"{t_rel:.6f}",
                int(round(t_rel * 1000)),
                f"{m['rss'] / BYTES_IN_MB:.2f}",
                is_duckdb, op_name, op_type, op_path, op_depth
            ])

    print("OK timeline ->", timeline_path)
    return 0

if __name__ == "__main__":
    main()