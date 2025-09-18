import os

import duckdb

from vtlengine.connection import con
from vtlengine.Exceptions import VtlEngineRemoteExtensionException

EXTRAS_DOCS = "https://docs.vtlengine.meaningfuldata.eu/#installation"
ERROR_MESSAGE = (
    "The '{extra_name}' extra is required to run {extra_desc}. "
    "Please install it using 'pip install vtlengine[{extra_name}]' or "
    "install all extras with 'pip install vtlengine[all]'. "
    f"Check the documentation at: {EXTRAS_DOCS}"
)


def __check_s3_extra() -> None:
    prod_mode = os.getenv("PROD_MODE", "False").lower() in ("1", "true", "yes")
    try:
        con.execute("LOAD httpfs;")
    except duckdb.Error:
        if prod_mode:
            raise VtlEngineRemoteExtensionException.remote_access_disabled()
        try:
            con.execute("INSTALL httpfs;")
            con.execute("LOAD httpfs;")
        except duckdb.Error:
            raise VtlEngineRemoteExtensionException.remote_access_disabled()
