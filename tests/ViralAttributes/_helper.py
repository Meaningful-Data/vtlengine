"""Shared file-based test helper for viral-attribute tests.

Viral tests follow the repo's standard convention: a VTL script under
``data/vtl/{code}.vtl`` plus DataStructure JSON and DataSet CSV files, executed
through :class:`tests.Helper.TestHelper`.
"""

from pathlib import Path

from tests.Helper import TestHelper


class ViralHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"
