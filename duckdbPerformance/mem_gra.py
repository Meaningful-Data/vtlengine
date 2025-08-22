import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("/Users/javierdiazcuadrado/Desktop/dukcdb/Binary/Numeric/DC/ML1GB/4GB/memory_timeline_2025-08-19_15-59-33.csv")
df = pd.read_csv(csv_path).sort_values("t_rel_s").reset_index(drop=True)

duration = df["t_rel_s"].max() - df["t_rel_s"].min()
peak_row = df.loc[df["rss_mb"].idxmax()]
peak_time = peak_row["t_rel_s"]
peak_mem = peak_row["rss_mb"]

print(f"Execution time: {duration:.2f} s")
print(f"Peak memory: {peak_mem:.2f} MB at t={peak_time:.2f} s")

sns.set_theme()
ax = sns.lineplot(data=df, x="t_rel_s", y="rss_mb", label="RSS (MB)")
ax.scatter(peak_time, peak_mem, color="red")

in_duckdb = False
start_t = None
start_rss = None
max_rss = None

for _, row in df.iterrows():
    if row["is_duckdb"] == 1 and not in_duckdb:
        in_duckdb = True
        start_t = row["t_rel_s"]
        start_rss = row["rss_mb"]
        max_rss = row["rss_mb"]
    elif row["is_duckdb"] == 1 and in_duckdb:
        if row["rss_mb"] > max_rss:
            max_rss = row["rss_mb"]
    elif row["is_duckdb"] == 0 and in_duckdb:
        end_t = row["t_rel_s"]
        mem_increase = max_rss - start_rss
        ax.axvspan(start_t, end_t, alpha=0.2, color="orange", label=f"DuckDB running (+{mem_increase:.2f} MB)")
        in_duckdb = False

if in_duckdb:
    end_t = df["t_rel_s"].iloc[-1]
    mem_increase = max_rss - start_rss
    ax.axvspan(start_t, end_t, alpha=0.2, color="orange", label=f"DuckDB running (+{mem_increase:.2f} MB)")

ax.text(peak_time, peak_mem, f"Peak {peak_mem:.2f} MB\nt= {peak_time:.2f}s",
        ha="left", va="bottom", fontsize=9, weight="bold")

ax.set_title("Memory evolution")
ax.set_xlabel("Relative time (s)")
ax.set_ylabel("Memory (MB)")
ax.legend()

out_path = csv_path.with_name(csv_path.stem + "_plot.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Graph saved at: {out_path}")

