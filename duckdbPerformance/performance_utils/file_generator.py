import json
import math
import os
import string
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

base_path = Path(__file__).parent.parent / "data"
BASIC_PATH = base_path / "BASIC_DATA"
BIG_PATH = base_path / "BIG_DATA"
DP_PATH = BIG_PATH / "dp"
DS_PATH = BIG_PATH / "ds"
BASE_LENGTH = int(1e6)
MAX_NUM = int(1e6)
MAX_STR = 12
ASCII_CHARS = 26
IDENTIFIER = "Identifier"
MEASURE = "Measure"


def ensure_dirs():
    for p in [BASIC_PATH, BIG_PATH, DP_PATH, DS_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def get_suffix(length: int) -> str:
    if length >= 1e9:
        return f"{int(length / 1e9)}G.csv"
    elif length >= 1e6:
        return f"{int(length / 1e6)}M.csv"
    elif length >= 1e3:
        return f"{int(length / 1e3)}K.csv"
    else:
        return f"{int(length)}.csv"


def generate_datastructure(dtypes: Dict[str, str], file_name: str):
    comps = {}
    file_name = file_name.replace(".csv", ".json")
    file_name = DS_PATH / file_name

    for column, dtype in dtypes.items():
        role = "Identifier" if column.lower().startswith("id") else "Measure"
        nullable = str(role == "Measure").lower()
        comps[column] = {
            "name": column,
            "data_type": dtype,
            "role": role,
            "nullable": nullable,
        }

    ds = {"datasets": [{"name": "DS_1", "DataStructure": list(comps.values())}]}

    with open(file_name, "w") as f:
        json.dump(ds, f, indent=4)


def int_to_str(n: int, length: int = 5) -> str:
    if n == 0:
        s = "A"
    else:
        s = ""
        while n > 0:
            s = chr(65 + (n % ASCII_CHARS)) + s
            n //= ASCII_CHARS
    if len(s) < length:
        s = "A" * (length - len(s)) + s
    return s


def get_min_str_length(length: int) -> int:
    return max(1, math.ceil(math.log(length, ASCII_CHARS)))


def generate_unique_combinations(identifiers, dtypes, length, max_vals, min_str_length, offset=0):
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
            data[col] = np.array([int_to_str(v, min_str_length) for v in val])
    return data


def generate_big_ass_csv(dtypes, length=None, chunk_size=1_000_000):
    ensure_dirs()

    length = int(length or BASE_LENGTH)
    identifiers = [col for col, dtype in dtypes.items() if col.lower().startswith("id")]
    measures = [col for col in dtypes if col not in identifiers]
    min_str_length = get_min_str_length(length)

    max_vals = []
    for col in identifiers:
        if dtypes[col] == "Integer":
            max_vals.append(MAX_NUM)
        elif dtypes[col] == "String":
            max_vals.append(ASCII_CHARS**min_str_length)
        else:
            raise ValueError(f"Unsupported identifier dtype: {dtypes[col]}")

    total_combos = np.prod(max_vals)
    if length > total_combos:
        raise ValueError(
            f"Cannot generate {length} unique rows with given identifiers. "
            f"Maximum possible: {total_combos}"
        )

    suffix = get_suffix(length)
    file_path = DP_PATH / f"BF_{suffix}"
    if file_path.exists():
        os.remove(file_path)

    generate_datastructure(dtypes, suffix)
    col_names = list(dtypes.keys())

    print(f"Creating csv '{file_path}' with {length:,} rows (chunks: {chunk_size:,}).")
    rows_written = 0
    while rows_written < length:
        cur_n = min(chunk_size, length - rows_written)
        unique_data = generate_unique_combinations(
            identifiers, dtypes, cur_n, max_vals, min_str_length, offset=rows_written
        )

        for col in measures:
            dt = dtypes[col]
            if dt == "Integer":
                unique_data[col] = np.random.randint(0, MAX_NUM, cur_n)
            elif dt == "Number":
                unique_data[col] = np.random.uniform(0, MAX_NUM, cur_n)
            elif dt == "Boolean":
                unique_data[col] = np.random.choice([True, False], cur_n)
            elif dt == "String":
                unique_data[col] = np.random.choice(
                    [
                        "".join(np.random.choice(list(string.ascii_letters), MAX_STR))
                        for _ in range(1000)
                    ],
                    cur_n,
                )
            else:
                raise ValueError(f"Unsupported dtype: {dt}")

        df = pd.DataFrame(unique_data, columns=col_names)
        df.to_csv(file_path, mode="a", index=False, header=(rows_written == 0))
        rows_written += cur_n
        print(f"  → Written: {rows_written:,} / {length:,} rows")

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)
    print(f"Generated CSV: '{file_path}' with {length:,} rows.")
    print(f"Final size: {size_mb:.2f} MB ({size_gb:.2f} GB)")


if __name__ == "__main__":
    print("Generating BIG_ASS CSV file")
    generate_big_ass_csv(
        dtypes={
            "Id_1": "Integer",
            "Id_2": "String",
            "Me_1": "Integer",
            "Me_2": "String",
            "Me_3": "Boolean",
        },
        length=2_000_000,
        chunk_size=2_000_000,
    )
