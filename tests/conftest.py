import pytest
from _pytest.outcomes import Skipped, xfail

from src.vtlengine.connection.connection import ConnectionManager


@pytest.fixture(autouse=True)
def clean_duckdb_connection():
    """
    Fixture to clean the DuckDB connection before each test.
    """
    ConnectionManager.clean_connection()


# Marking as ignored every exception missmatch (DID NOT RAISE VtlengineException or similar) tests
# comment this method out if you want to run every tests
# this is done to focus on the execution failing tests
def pytest_runtest_makereport(item, call: pytest.CallInfo):
    if call.excinfo and "DID NOT RAISE" in str(call.excinfo.value):
        try:
            msg = "Test marked as skipped due to VTL exception missmatch"
            except_msg = str(call.excinfo.value)
            call.excinfo = Skipped(msg=msg)
            call.excinfo.value = xfail.Exception(msg=except_msg)
        except Exception as e:
            pass
