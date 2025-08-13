import os
import json
import random
import string
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd



base_path = Path(__file__).parent / "data"
BASIC_PATH = base_path / "BASIC_DATA"
BIG_PATH = base_path / "BIG_DATA"
DP_PATH = BIG_PATH / "dp"
DS_PATH = BIG_PATH / "ds"
BASE_LENGTH = int(1e6)
MAX_NUM = int(1e6)
MAX_STR = 20
IDENTIFIER = "Identifier"
MEASURE = "Measure"

def string_stream(max_len: int):
    letters = string.ascii_uppercase
    while True:
        for l in range(1, max_len + 1):
            pow_l = 26 ** l
            for s in range(pow_l):
                yield ''.join(letters[(s // (26 ** i)) % 26] for i in range(l - 1, -1, -1))


def ensure_dirs():
    DP_PATH.mkdir(parents=True, exist_ok=True)
    DS_PATH.mkdir(parents=True, exist_ok=True)


def get_suffix(length: int) -> str:
    if length >= 1_000_000_000:
        return f"{int(length / 1e9)}G.csv"
    elif length >= 1_000_000:
        return f"{int(length / 1e6)}M.csv"
    elif length >= 1_000:
        return f"{int(length / 1e3)}K.csv"
    else:
        return f"{int(length)}.csv"


def generate_datastructure(dtypes, file_name):
    comps = {}
    file_name = file_name.replace(".csv", ".json")
    file_name = DS_PATH / file_name

    for column, dtype in dtypes.items():
        role = IDENTIFIER if column.lower().startswith("id") else MEASURE
        nullable = str(role == MEASURE).lower() == "true"
        role = role.capitalize()
        vtl_dtype = dtype

        comps[column] = {
            "name": column,
            "type": vtl_dtype,
            "role": role,
            "nullable": nullable,
        }

    ds = {
        "datasets": [
            {"name": "DS_1", "DataStructure": [comp for comp in comps.values()]}
        ]
    }

    with open(file_name, "w") as f:
        json.dump(ds, f, indent=4)


def reverse_cast_mapping(python_type):
    cast_mapping = {
        "String": str,
        "Number": float,
        "Integer": int,
        "TimeInterval": str,
        "Date": str,
        "TimePeriod": str,
        "Duration": str,
        "Boolean": bool,
    }
    reversed_mapping = {v: k for k, v in cast_mapping.items()}
    return reversed_mapping.get(python_type, "Null")


def generate_big_ass_csv(dtypes: Dict[str, str], length: int = None, chunk_size: int = 1_000_000):
    if not dtypes:
        raise Exception("Need to pass dtype")
    ensure_dirs()

    length = int(length or BASE_LENGTH)
    if length > 12_000_000:
        possible_str_number = 7
    elif length > 5_000_000:
        possible_str_number = 6
    elif length > 470_000:
        possible_str_number = 5
    elif length > 18_000:
        possible_str_number = 4
    else:
        possible_str_number = 3

    suffix = get_suffix(length)
    generate_datastructure(dtypes, suffix)

    file_path = DP_PATH / f"BF_{suffix}"
    header_written = False

    string_generators = {
        col: string_stream(possible_str_number)
        for col, dt in dtypes.items() if dt == "String"
    }

    rows_written = 0
    col_names = list(dtypes.keys())

    print(f"Creating csv '{file_path}' with {length:,} rows (chunks: {chunk_size:,}).")

    while rows_written < length:
        cur_n = min(chunk_size, length - rows_written)
        data = {}

        for col, dt in dtypes.items():
            if dt == "Integer":
                data[col] = np.random.randint(0, MAX_NUM, size=cur_n, dtype=np.int64)
            elif dt == "Number":
                data[col] = np.random.uniform(0, MAX_NUM, size=cur_n)
            elif dt == "Boolean":
                data[col] = np.random.randint(0, 2, size=cur_n, dtype=bool)
            elif dt == "String":
                gen = string_generators[col]
                data[col] = [next(gen) for _ in range(cur_n)]
            else:
                raise ValueError(f"Unsupported dtype: {dt}")

        df = pd.DataFrame(data, columns=col_names)
        df.fillna("", inplace=True)
        df.fillna(0, inplace=True)
        df.to_csv(file_path, mode='a', index=False, header=not header_written)
        header_written = True

        rows_written += cur_n
        print(f"  → Writen: {rows_written:,} / {length:,} rows")

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 ** 2)
    size_gb = size_bytes / (1024 ** 3)

    print(f"Generated CSV: '{file_path}' with {length:,} rows.")
    print(f"Final size: {size_mb:.2f} MB ({size_gb:.2f} GB)")


if __name__ == "__main__":
    print("Generando archivo de ~10GB")
    generate_big_ass_csv(
        dtypes={"Id_1": "Integer", "Id_2": "String", "Me_1": "Integer", "Me_2": "String", "Me_3": "Boolean"},
        length=2_000_000,
        chunk_size=2_000_000
    )
