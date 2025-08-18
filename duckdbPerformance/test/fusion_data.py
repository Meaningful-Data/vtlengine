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

def node_duration(n, dflt=0.0):
    dur = seconds(n.get("operator_timing", 0.0))
    if dur == 0.0 and n.get("children"):
        s = sum(node_duration(c, 0.0) for c in n["children"])
        dur = s if s > 0 else 0.0
    return dur if dur > 0 else dflt

def flatten_windows(node, end_perf, path=None, depth=0):
    path = path or []
    name = node.get("operator_name", node.get("query_name", "ROOT"))
    typ  = node.get("operator_type", "ROOT")
    dur  = node_duration(node, dflt=seconds(node.get("latency", 0.0)))
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
        cur_end = end_perf
        for ch in reversed(children):
            rows += flatten_windows(ch, cur_end, path + [name], depth + 1)
            cur_end -= node_duration(ch, 0.0)
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
        print("output/finish.json not found")
        return 1

    profile = OUT_DIR / "logs" / "logs.json"
    if not profile.exists():
        print("output/logs.json not found")
        return 1

    series = read_mem_series(mem_csv)
    prof   = load_json(profile)
    fin    = load_json(finish)

    if "perf_end" not in fin:
        print("finish.json is missing 'perf_end'")
        return 1

    perf_end = float(fin["perf_end"])
    latency  = seconds(prof.get("latency", 0.0)) or node_duration({"children": prof.get("children", [])}, 0.0)
    if latency <= 0.0:
        print("Could not determine total DuckDB latency")
        return 1

    perf_start = perf_end - latency

    root = {
        "operator_name": prof.get("query_name", "DUCKDB_QUERY"),
        "operator_type": "ROOT",
        "operator_timing": latency,
        "children": prof.get("children", []),
        "latency": latency,
    }
    windows = flatten_windows(root, perf_end)
    windows[0]["start"] = perf_start
    windows[0]["end"]   = perf_end
    windows[0]["dur"]   = latency

    timeline_path = OUT_DIR / f"memory_timeline_{run_id}.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["perf_s","perf_ms","t_rel_s","t_rel_ms","rss_mb","is_duckdb","op_name","op_type","op_path","op_depth"])
        for m in series:
            op = active_op_at(m["perf"], windows)
            if op:
                is_duckdb = 1
                op_name = "DUCKDB_QUERY" if op["type"] == "ROOT" else op["name"]
                op_type = op["type"]; op_path = op["path"]; op_depth = op["depth"]
            else:
                is_duckdb = 0
                op_name = op_type = op_path = ""; op_depth = ""
            w.writerow([
                f"{m['perf']:.6f}",
                int(round(m["perf"] * 1000)),
                f"{m['t_rel_s']:.6f}",
                int(round(m["t_rel_s"] * 1000)),
                f"{m['rss'] / BYTES_IN_MB:.2f}",
                is_duckdb, op_name, op_type, op_path, op_depth
            ])

    print("OK timeline ->", timeline_path)
    return 0

if __name__ == "__main__":
    main()