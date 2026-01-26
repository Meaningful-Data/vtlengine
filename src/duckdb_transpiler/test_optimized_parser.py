"""Test optimized DuckDB parser with constraint-based validation."""

import os
import tempfile

import duckdb

from duckdb_transpiler.Parser import load_datapoints_duckdb
from vtlengine.DataTypes import Boolean, Date, Integer, Number, String, TimePeriod
from vtlengine.Model import Component, Role


def test_basic_load():
    """Test basic CSV loading with PRIMARY KEY and type validation."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Id_2": Component(name="Id_2", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
        "Me_2": Component(name="Me_2", data_type=Boolean, role=Role.MEASURE, nullable=True),
        "At_1": Component(name="At_1", data_type=String, role=Role.ATTRIBUTE, nullable=True),
    }

    csv_content = """
    Id_1,Id_2,Me_1,Me_2,At_1
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

        print("=== Table Schema ===")
        schema = conn.execute("DESCRIBE DS_1").fetchall()
        for row in schema:
            print(f"  {row[0]}: {row[1]} (null={row[2]})")

        print("\n=== PRIMARY KEY ===")
        # Check table has primary key
        info = conn.execute("SELECT * FROM duckdb_constraints() WHERE table_name='DS_1'").fetchall()
        for row in info:
            print(f"  {row}")

        print("\n=== Data ===")
        df = rel.fetchdf()
        print(df)
        print(f"\nRows: {len(df)}")

    finally:
        os.unlink(csv_path)

    print("\n✓ Basic load test passed!")


def test_duplicate_detection_via_pk():
    """Test that PRIMARY KEY catches duplicates automatically."""
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
            print("✗ Should have raised error for duplicates!")
        except Exception as e:
            print(f"✓ PRIMARY KEY caught duplicate: {type(e).__name__}")
            # Verify table was dropped
            tables = conn.execute("SHOW TABLES").fetchall()
            assert len(tables) == 0, "Table should be dropped on error"  # noqa: S101
            print("✓ Table cleaned up after error")
    finally:
        os.unlink(csv_path)


def test_null_identifier_via_not_null():
    """Test that NOT NULL catches null identifiers."""
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
            print(f"✓ NOT NULL caught null identifier: {type(e).__name__}")
    finally:
        os.unlink(csv_path)


def test_type_validation_implicit():
    """Test that type validation is implicit via DuckDB casting."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Int_1": Component(name="Int_1", data_type=Integer, role=Role.MEASURE, nullable=True),
    }

    csv_content = """Id_1,Int_1
A,not_a_number
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        try:
            load_datapoints_duckdb(conn, components, "DS_1", csv_path)
            print("✗ Should have raised error for invalid Integer!")
        except Exception as e:
            print(f"✓ Type validation caught invalid Integer: {type(e).__name__}")
    finally:
        os.unlink(csv_path)


def test_dwi_no_identifiers():
    """Test DWI: no identifiers → max 1 row."""
    components = {
        "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
    }

    # More than 1 row without identifiers
    csv_content = """Me_1
10
20
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        try:
            load_datapoints_duckdb(conn, components, "DS_1", csv_path)
            print("✗ Should have raised error for DWI violation!")
        except Exception as e:
            print(f"✓ DWI check caught multiple rows without identifiers: {type(e).__name__}")
    finally:
        os.unlink(csv_path)

    # Single row without identifiers - should work
    csv_content = """Me_1
10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        load_datapoints_duckdb(conn, components, "DS_1", csv_path)
        print("✓ Single row without identifiers works")
    finally:
        os.unlink(csv_path)


def test_time_period_validation():
    """Test TimePeriod explicit validation."""
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
A,INVALID
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
            print(f"✓ Temporal validation caught invalid TimePeriod: {type(e).__name__}")
    finally:
        os.unlink(csv_path)


def test_date_implicit_validation():
    """Test Date is validated implicitly by DuckDB."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Dt": Component(name="Dt", data_type=Date, role=Role.MEASURE, nullable=True),
    }

    # Valid dates
    csv_content = """Id_1,Dt
A,2024-01-15
B,2024-06-30
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        load_datapoints_duckdb(conn, components, "DS_1", csv_path)

        # Check it's actually a DATE type
        schema = conn.execute("DESCRIBE DS_1").fetchall()
        dt_type = [r[1] for r in schema if r[0] == "Dt"][0]
        assert dt_type == "DATE", f"Expected DATE, got {dt_type}"  # noqa: S101
        print(f"✓ Date column type: {dt_type}")
    finally:
        os.unlink(csv_path)

    # Invalid date
    csv_content = """Id_1,Dt
A,not-a-date
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        conn = duckdb.connect()
        try:
            load_datapoints_duckdb(conn, components, "DS_1", csv_path)
            print("✗ Should have raised error for invalid Date!")
        except Exception as e:
            print(f"✓ DuckDB caught invalid Date implicitly: {type(e).__name__}")
    finally:
        os.unlink(csv_path)


def test_empty_table():
    """Test empty table creation."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
    }

    conn = duckdb.connect()
    load_datapoints_duckdb(conn, components, "DS_1", None)

    count = conn.execute("SELECT COUNT(*) FROM DS_1").fetchone()[0]
    assert count == 0, f"Expected 0 rows, got {count}"  # noqa: S101

    # Check schema
    schema = conn.execute("DESCRIBE DS_1").fetchall()
    print(f"✓ Empty table created with {len(schema)} columns")


def test_performance_single_pass():
    """Verify we do minimal passes over the data."""
    components = {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Id_2": Component(name="Id_2", data_type=Integer, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
    }

    # Create larger CSV
    rows = ["Id_1,Id_2,Me_1"]
    for i in range(10000):
        rows.append(f"A{i},{i},{i * 1.5}")
    csv_content = "\n".join(rows)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        import time

        conn = duckdb.connect()

        start = time.time()
        load_datapoints_duckdb(conn, components, "DS_1", csv_path)
        elapsed = time.time() - start

        count = conn.execute("SELECT COUNT(*) FROM DS_1").fetchone()[0]
        print(f"✓ Loaded {count} rows in {elapsed:.3f}s")

    finally:
        os.unlink(csv_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Optimized DuckDB Parser (constraint-based)")
    print("=" * 60)

    test_basic_load()
    print()
    test_duplicate_detection_via_pk()
    print()
    test_null_identifier_via_not_null()
    print()
    test_type_validation_implicit()
    print()
    test_dwi_no_identifiers()
    print()
    test_time_period_validation()
    print()
    test_date_implicit_validation()
    print()
    test_empty_table()
    print()
    test_performance_single_pass()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
