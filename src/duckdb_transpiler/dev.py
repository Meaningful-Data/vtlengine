import tempfile
from pathlib import Path

import pandas as pd

from duckdb_transpiler.API import run, transpile


def test_transpile():
    """Test transpile function - generates SQL without execution."""
    print("\n" + "=" * 70)
    print("TESTING TRANSPILE (SQL Generation)")
    print("=" * 70)

    script = """
        // Dataset-Dataset operation
        DS_r1 <- DS_1 + DS_2;

        // Dataset-Scalar operation
        DS_r2 <- DS_1 * 10;

        // Clause: calc with new column
        DS_r3 <- DS_1[calc Me_1_copy := Me_1];

        // Clause: filter
        DS_r4 <- DS_1[filter Me_1 > 10];

        // Clause: keep
        DS_r5 <- DS_1[keep Me_1];

        // Scalar-Scalar operation
        sc_r <- 2 * 3 + 1;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False},
                    {"name": "Id_2", "role": "Identifier", "type": "String", "nullable": False},
                    {"name": "Me_1", "role": "Measure", "type": "Number", "nullable": True},
                ],
            },
            {
                "name": "DS_2",
                "DataStructure": [
                    {"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False},
                    {"name": "Id_2", "role": "Identifier", "type": "String", "nullable": False},
                    {"name": "Me_1", "role": "Measure", "type": "Number", "nullable": True},
                ],
            },
        ],
        "scalars": [{"name": "sc_1", "type": "Integer"}],
    }

    queries = transpile(script=script, data_structures=data_structures)

    for name, sql, is_persistent in queries:
        print(f"\n--- {name} (persistent={is_persistent}) ---")
        print(sql)


def test_run_with_dataframes():
    """Test run function with DataFrame inputs."""
    print("\n" + "=" * 70)
    print("TESTING RUN (Full Execution with DataFrames)")
    print("=" * 70)

    script = """
        // Simple arithmetic
        DS_r1 <- DS_1 + DS_2;

        // Calc with new column
        DS_r2 <- DS_1[calc Me_doubled := Me_1 * 2];

        // Filter
        DS_r3 <- DS_1[filter Me_1 > 15];
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False},
                    {"name": "Id_2", "role": "Identifier", "type": "String", "nullable": False},
                    {"name": "Me_1", "role": "Measure", "type": "Number", "nullable": True},
                ],
            },
            {
                "name": "DS_2",
                "DataStructure": [
                    {"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False},
                    {"name": "Id_2", "role": "Identifier", "type": "String", "nullable": False},
                    {"name": "Me_1", "role": "Measure", "type": "Number", "nullable": True},
                ],
            },
        ],
    }

    # Create test data as DataFrames
    df1 = pd.DataFrame(
        {
            "Id_1": [1, 2, 3],
            "Id_2": ["A", "B", "C"],
            "Me_1": [10.5, 20.5, 30.5],
        }
    )

    df2 = pd.DataFrame(
        {
            "Id_1": [1, 2, 3],
            "Id_2": ["A", "B", "C"],
            "Me_1": [1.0, 2.0, 3.0],
        }
    )

    datapoints = {"DS_1": df1, "DS_2": df2}

    results = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
    )

    for name, result in results.items():
        print(f"\n--- {name} ---")
        print(result)


def test_run_with_csv():
    """Test run function with CSV file inputs."""
    print("\n" + "=" * 70)
    print("TESTING RUN (Full Execution with CSV Files)")
    print("=" * 70)

    script = """
        DS_r1 <- DS_1 * 2;
        DS_r2 <- DS_1[calc Me_doubled := Me_1 * 2];
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False},
                    {"name": "Id_2", "role": "Identifier", "type": "String", "nullable": False},
                    {"name": "Me_1", "role": "Measure", "type": "Number", "nullable": True},
                ],
            },
        ],
    }

    # Create temporary CSV file
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "DS_1.csv"
        csv_content = """
        Id_1,Id_2,Me_1
        1,A,10.5
        2,B,20.5
        3,C,30.5
        """
        csv_path.write_text(csv_content)

        results = run(
            script=script,
            data_structures=data_structures,
            datapoints={"DS_1": csv_path},
        )

        for name, result in results.items():
            print(f"\n--- {name} ---")
            print(result)


def test_type_casting():
    """Test type casting with various data types."""
    print("\n" + "=" * 70)
    print("TESTING TYPE CASTING")
    print("=" * 70)

    script = """
        DS_r <- DS_1;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "role": "Identifier", "type": "Integer", "nullable": False},
                    {"name": "Str_col", "role": "Measure", "type": "String", "nullable": True},
                    {"name": "Int_col", "role": "Measure", "type": "Integer", "nullable": True},
                    {"name": "Num_col", "role": "Measure", "type": "Number", "nullable": True},
                    {"name": "Bool_col", "role": "Measure", "type": "Boolean", "nullable": True},
                    {"name": "Date_col", "role": "Measure", "type": "Date", "nullable": True},
                ],
            },
        ],
    }

    # Create temp CSV with various types
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "DS_1.csv"
        # Test: "2.0" -> 2 for Integer, "true"/"1" -> True for Boolean
        csv_content = """Id_1,Str_col,Int_col,Num_col,Bool_col,Date_col
1,hello,2.0,3.14159,true,2024-01-15
2,world,42,2.71828,1,2024-06-30
3,test,100,0.5,false,2024-12-31
"""
        csv_path.write_text(csv_content)

        results = run(
            script=script,
            data_structures=data_structures,
            datapoints={"DS_1": csv_path},
        )

        for name, result in results.items():
            print(f"\n--- {name} ---")
            print(result)
            print(f"\nData types:\n{result.dtypes}")


if __name__ == "__main__":
    test_transpile()
    test_run_with_dataframes()
    test_run_with_csv()
    test_type_casting()
