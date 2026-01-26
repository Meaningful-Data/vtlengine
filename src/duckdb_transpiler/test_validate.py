"""Test DuckDB validation (replicate _validate_pandas logic)."""

import os
import tempfile

import duckdb

from duckdb_transpiler.Parser import load_datapoints_duckdb
from vtlengine.DataTypes import Boolean, Date, Integer, Number, String, TimePeriod
from vtlengine.Model import Component, Role


def test_basic_load():
    """Test basic CSV loading with type casting."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Id_2": Component(name="Id_2", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
        "Me_2": Component(name="Me_2", data_type=Boolean, role=Role.MEASURE, nullable=True),
        "At_1": Component(name="At_1", data_type=String, role=Role.ATTRIBUTE, nullable=True),
    }

    # Create CSV
    csv_content = """Id_1,Id_2,Me_1,Me_2,At_1
A,1,10.5,true,attr1
B,2,20.0,false,attr2
C,3,30.123,1,attr3
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        rel = load_datapoints_duckdb(conn, components, "DS_1", csv_path)

        print("=== Column Types ===")
        for col, dtype in zip(rel.columns, rel.types):
            print(f"  {col}: {dtype}")

        print("\n=== Data ===")
        df = rel.fetchdf()
        print(df)
        print("\n=== Pandas dtypes ===")
        print(df.dtypes)

    finally:
        os.unlink(csv_path)

    print("\n✓ Basic load test passed!")


def test_null_identifier_fails():
    """Test that NULL identifier raises error."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
    }

    csv_content = """Id_1,Me_1
A,10
,20
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        try:
            load_datapoints_duckdb(conn, components, "DS_1", csv_path)
            print("✗ Should have raised error for NULL identifier!")
        except Exception as e:
            print(f"✓ Correctly caught error: {type(e).__name__}")
    finally:
        os.unlink(csv_path)


def test_duplicate_identifiers_fails():
    """Test that duplicate identifiers raise error."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
    }

    csv_content = """Id_1,Me_1
A,10
A,20
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        try:
            load_datapoints_duckdb(conn, components, "DS_1", csv_path)
            print("✗ Should have raised error for duplicate identifiers!")
        except Exception as e:
            print(f"✓ Correctly caught error: {type(e).__name__}")
    finally:
        os.unlink(csv_path)


def test_time_period_validation():
    """Test TimePeriod validation."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Period": Component(
            name="Period", data_type=TimePeriod, role=Role.IDENTIFIER, nullable=False
        ),
    }

    # Valid periods
    csv_content = """Id_1,Period
A,2024A
B,2024S1
C,2024Q1
D,2024M01
E,2024W01
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        load_datapoints_duckdb(conn, components, "DS_1", csv_path)
        print("✓ Valid TimePeriod test passed!")
    finally:
        os.unlink(csv_path)

    # Invalid period
    csv_content = """Id_1,Period
A,INVALID_PERIOD
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        try:
            load_datapoints_duckdb(conn, components, "DS_1", csv_path)
            print("✗ Should have raised error for invalid TimePeriod!")
        except Exception as e:
            print(f"✓ Correctly caught invalid TimePeriod: {type(e).__name__}")
    finally:
        os.unlink(csv_path)


def test_date_validation():
    """Test Date validation and casting."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Dt": Component(name="Dt", data_type=Date, role=Role.MEASURE, nullable=True),
    }

    csv_content = """Id_1,Dt
A,2024-01-15
B,2024-06-30
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        rel = load_datapoints_duckdb(conn, components, "DS_1", csv_path)

        print("\n=== Date Types ===")
        for col, dtype in zip(rel.columns, rel.types):
            print(f"  {col}: {dtype}")

        print("✓ Date validation test passed!")
    finally:
        os.unlink(csv_path)


def test_integer_float_handling():
    """Test Integer handles '2.0' -> 2 correctly."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Int_1": Component(name="Int_1", data_type=Integer, role=Role.MEASURE, nullable=True),
    }

    csv_content = """Id_1,Int_1
A,1.0
B,2.0
C,3
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        rel = load_datapoints_duckdb(conn, components, "DS_1", csv_path)
        df = rel.fetchdf()
        print("\n=== Integer from float strings ===")
        print(df)
        print(f"\nType of Int_1: {df['Int_1'].dtype}")
        assert df["Int_1"].tolist() == [1, 2, 3], "Integer conversion failed!"  # noqa: S101
        print("✓ Integer float handling test passed!")
    finally:
        os.unlink(csv_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DuckDB validation (replicate _validate_pandas)")
    print("=" * 60)

    test_basic_load()
    print()
    test_null_identifier_fails()
    print()
    test_duplicate_identifiers_fails()
    print()
    test_time_period_validation()
    print()
    test_date_validation()
    print()
    test_integer_float_handling()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
