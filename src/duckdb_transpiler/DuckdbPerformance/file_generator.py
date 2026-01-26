"""
High-performance CSV file generator for testing DuckDB parser.

Generates large CSV files with unique identifiers and random measures.
Supports both Python (optimized with NumPy) and C implementations.
"""

import contextlib
import json
import math
import os
import string
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

BASE_PATH = Path(__file__).parent.parent / "data"
BIG_PATH = BASE_PATH / "BIG_DATA"
DP_PATH = BIG_PATH / "dp"
DS_PATH = BIG_PATH / "ds"

BASE_LENGTH = int(1e6)
MAX_NUM = int(1e6)
MAX_STR = 12
ASCII_CHARS = 26

IDENTIFIER = "Identifier"
MEASURE = "Measure"

C_BIN_NAME = "file_generator"


# =============================================================================
# Utility Functions
# =============================================================================


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for p in [BIG_PATH, DP_PATH, DS_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def get_suffix(length: int) -> str:
    """Get human-readable size suffix for filename."""
    if length >= 1e9:
        return f"{int(length / 1e9)}G.csv"
    elif length >= 1e6:
        return f"{int(length / 1e6)}M.csv"
    elif length >= 1e3:
        return f"{int(length / 1e3)}K.csv"
    else:
        return f"{int(length)}.csv"


def generate_datastructure(dtypes: Dict[str, str], file_name: str) -> Path:
    """
    Generate VTL data structure JSON file.

    Args:
        dtypes: Column name → VTL type mapping
        file_name: CSV filename (will be converted to .json)

    Returns:
        Path to generated JSON file
    """
    json_name = file_name.replace(".csv", ".json")
    json_path = DS_PATH / json_name

    comps = []
    for column, dtype in dtypes.items():
        role = IDENTIFIER if column.lower().startswith("id") else MEASURE
        comps.append(
            {
                "name": column,
                "type": dtype,
                "role": role,
                "nullable": role == MEASURE,
            }
        )

    ds = {"datasets": [{"name": "DS_1", "DataStructure": comps}]}

    with open(json_path, "w") as f:
        json.dump(ds, f, indent=4)

    return json_path


# =============================================================================
# ID Generation Helpers
# =============================================================================


def int_to_str_fast(arr: np.ndarray, width: int) -> np.ndarray:
    """
    Ultra-fast vectorized int-to-string conversion (A-Z encoding).

    Uses modular arithmetic to convert integers to fixed-width strings.
    """
    n = len(arr)
    result = np.empty((n, width), dtype="U1")

    for pos in range(width):
        divisor = ASCII_CHARS ** (width - 1 - pos)
        digit = (arr // divisor) % ASCII_CHARS
        result[:, pos] = [chr(65 + d) for d in digit]

    return np.array(["".join(row) for row in result])


def get_min_str_length(length: int) -> int:
    """Calculate minimum string length to represent `length` unique values."""
    return max(1, math.ceil(math.log(max(1, length), ASCII_CHARS)))


# =============================================================================
# Python CSV Generator
# =============================================================================


def generate_unique_ids(
    identifiers: list,
    dtypes: Dict[str, str],
    length: int,
    max_vals: list,
    min_str_length: int,
    offset: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Generate unique identifier combinations using modular arithmetic.

    Uses a mixed-radix number system to generate unique combinations.
    """
    factors = []
    prod = 1
    for base in reversed(max_vals):
        factors.insert(0, prod)
        prod *= base

    idx = np.arange(offset, offset + length, dtype=np.int64)

    data = {}
    for i, col in enumerate(identifiers):
        base = max_vals[i]
        val = (idx // factors[i]) % base

        if dtypes[col] == "Integer":
            data[col] = val
        elif dtypes[col] == "String":
            data[col] = int_to_str_fast(val, min_str_length)

    return data


def generate_unique_ids_shuffled(
    identifiers: list,
    dtypes: Dict[str, str],
    indices: np.ndarray,
    max_vals: list,
    min_str_length: int,
) -> Dict[str, np.ndarray]:
    """Generate identifiers for specific (potentially non-sequential) indices."""
    factors = []
    prod = 1
    for base in reversed(max_vals):
        factors.insert(0, prod)
        prod *= base

    data = {}
    for i, col in enumerate(identifiers):
        base = max_vals[i]
        val = (indices // factors[i]) % base

        if dtypes[col] == "Integer":
            data[col] = val
        elif dtypes[col] == "String":
            data[col] = int_to_str_fast(val, min_str_length)

    return data


def generate_big_csv(
    dtypes: Dict[str, str],
    length: Optional[int] = None,
    chunk_size: int = 1_000_000,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Path:
    """
    Generate large CSV with unique identifiers and random measures.

    Args:
        dtypes: Column name → VTL type mapping
        length: Number of rows (default: 1M)
        chunk_size: Rows per chunk for memory efficiency
        shuffle: If True, randomize row order
        seed: Random seed for reproducibility

    Returns:
        Path to generated CSV file
    """
    ensure_dirs()
    rng = np.random.default_rng(seed)

    length = int(length or BASE_LENGTH)
    identifiers = [col for col in dtypes if col.lower().startswith("id")]
    measures = [col for col in dtypes if col not in identifiers]
    min_str_len = get_min_str_length(length)

    # Calculate max values per identifier
    max_vals = []
    for col in identifiers:
        if dtypes[col] == "Integer":
            max_vals.append(MAX_NUM)
        elif dtypes[col] == "String":
            max_vals.append(ASCII_CHARS**min_str_len)
        else:
            raise ValueError(f"Unsupported identifier dtype: {dtypes[col]}")

    # Verify we can generate enough unique combinations
    total_combos = math.prod(max_vals) if max_vals else 1
    if length > total_combos:
        raise ValueError(f"Cannot generate {length:,} unique rows. Max: {total_combos:,}")

    # Setup file paths
    suffix = get_suffix(length)
    file_path = DP_PATH / f"BF_{suffix}"
    if file_path.exists():
        os.remove(file_path)

    generate_datastructure(dtypes, suffix)
    col_names = list(dtypes.keys())

    print(f"Generating CSV (Python): '{file_path}'")
    print(f"  Rows: {length:,} | Chunks: {chunk_size:,} | Shuffle: {shuffle}")

    # For shuffle: generate all indices first and shuffle them
    all_indices = rng.permutation(length) if shuffle else None

    rows_written = 0
    start_time = time.time()

    while rows_written < length:
        cur_n = min(chunk_size, length - rows_written)

        # Generate identifiers
        if shuffle:
            chunk_indices = all_indices[rows_written : rows_written + cur_n]
            unique_data = generate_unique_ids_shuffled(
                identifiers, dtypes, chunk_indices, max_vals, min_str_len
            )
        else:
            unique_data = generate_unique_ids(
                identifiers, dtypes, cur_n, max_vals, min_str_len, offset=rows_written
            )

        # Generate measures (random data)
        for col in measures:
            dt = dtypes[col]
            if dt == "Integer":
                unique_data[col] = rng.integers(0, MAX_NUM, cur_n)
            elif dt == "Number":
                unique_data[col] = rng.uniform(0, MAX_NUM, cur_n)
            elif dt == "Boolean":
                unique_data[col] = rng.choice([True, False], cur_n)
            elif dt == "String":
                pool = [
                    "".join(rng.choice(list(string.ascii_letters), MAX_STR))
                    for _ in range(min(1000, cur_n))
                ]
                unique_data[col] = rng.choice(pool, cur_n)
            else:
                raise ValueError(f"Unsupported measure dtype: {dt}")

        df = pd.DataFrame(unique_data, columns=col_names)
        df.to_csv(file_path, mode="a", index=False, header=(rows_written == 0))
        rows_written += cur_n

        elapsed = time.time() - start_time
        rate = rows_written / elapsed if elapsed > 0 else 0
        print(f"  → {rows_written:,} / {length:,} rows ({rate:,.0f} rows/s)")

    elapsed = time.time() - start_time
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024**2)
    print(f"✓ Generated: {file_path}")
    print(f"  Size: {size_mb:.2f} MB | Time: {elapsed:.2f}s | Rate: {length / elapsed:,.0f} rows/s")

    return file_path


# =============================================================================
# C CSV Generator (Fast)
# =============================================================================


def generate_big_csv_fast(
    dtypes: Dict[str, str],
    length: Optional[int] = None,
    chunk_size: int = 1_000_000,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Path:
    """
    Generate large CSV using optimized C implementation.

    Much faster than Python version for very large files.

    Args:
        dtypes: Column name → VTL type mapping
        length: Number of rows (default: 1M)
        chunk_size: Rows per chunk
        shuffle: If True, randomize row order
        seed: Random seed for reproducibility

    Returns:
        Path to generated CSV file
    """
    ensure_dirs()
    length = int(length or BASE_LENGTH)
    suffix = get_suffix(length)
    csv_path = DP_PATH / f"BF_{suffix}"
    ds_json_path = DS_PATH / suffix.replace(".csv", ".json")

    if csv_path.exists():
        os.remove(csv_path)

    generate_datastructure(dtypes, suffix)

    # Build config file for C program
    config_lines = [
        f"rows={length}",
        f"chunk_size={chunk_size}",
        f"max_num={MAX_NUM}",
        f"max_str={MAX_STR}",
        f"csv_path={csv_path}",
        f"ds_path={ds_json_path}",
        f"columns={len(dtypes)}",
        f"shuffle={1 if shuffle else 0}",
        f"seed={seed if seed is not None else int(time.time())}",
    ]

    for name, dt in dtypes.items():
        config_lines.append(f"col={name}|{dt}")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".cfg") as cfg:
        cfg.write("\n".join(config_lines))
        cfg_path = cfg.name

    try:
        bin_path = Path(__file__).parent / C_BIN_NAME

        if not bin_path.exists():
            print(f"C binary not found at {bin_path}")
            print("Run 'make' in the DuckdbPerformance directory to compile it.")
            raise FileNotFoundError(f"C binary not found: {bin_path}")

        print(f"Generating CSV (C): '{csv_path}'")
        print(f"  Rows: {length:,} | Chunks: {chunk_size:,} | Shuffle: {shuffle}")

        start_time = time.time()
        subprocess.check_call([str(bin_path), cfg_path])  # noqa: S603
        elapsed = time.time() - start_time

        size_bytes = os.path.getsize(csv_path)
        size_mb = size_bytes / (1024**2)
        print(f"✓ Generated: {csv_path}")
        print(
            f"  Size: {size_mb:.2f} MB | Time: {elapsed:.2f}s | "
            f"Rate: {length / elapsed:,.0f} rows/s"
        )

        return csv_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"C generator failed with exit code {e.returncode}") from e
    except FileNotFoundError:
        raise
    except Exception as e:
        raise OSError(f"Failed to run C generator: {e}") from e
    finally:
        with contextlib.suppress(OSError):
            os.remove(cfg_path)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate large CSV files for testing")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Number of rows")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Chunk size")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--python", action="store_true", help="Use Python generator (slower)")
    args = parser.parse_args()

    dtypes = {
        "Id_1": "Integer",
        "Id_2": "String",
        "Id_3": "Integer",
        "Me_1": "Number",
        "Me_2": "Integer",
    }

    if args.python:
        generate_big_csv(
            dtypes=dtypes,
            length=args.rows,
            chunk_size=args.chunk_size,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    else:
        generate_big_csv_fast(
            dtypes=dtypes,
            length=args.rows,
            chunk_size=args.chunk_size,
            shuffle=args.shuffle,
            seed=args.seed,
        )
