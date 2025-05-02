from typing import Any

import duckdb

con = duckdb.connect(database=":memory:", read_only=False)
con.execute("SET memory_limit='512MB';")
con.execute("SET enable_progress_bar = true;")


VTL_TO_SQL_TYPE_MAP = {
    "String": "VARCHAR",
    "Integer": "INTEGER",
    "Number": "FLOAT",
    "Boolean": "BOOLEAN",
    "Date": "DATE",
    "TimePeriod": "DATE",
    "Time": "TIME",
    "Duration": "INTERVAL",
    "Null": "NULL",
}


BIN_OP_SQL_TOKENS = {
    "==": "=",
    "!=": "<>",
}


def get_sql_type(vtl_type: str) -> str:
    """
    Convert VTL type to SQL type.
    """
    return VTL_TO_SQL_TYPE_MAP.get(vtl_type, "VARCHAR")


def get_vtl_type(sql_type: str) -> str:
    """
    Convert SQL type to VTL type.
    """
    for vtl_type, sql_type_value in VTL_TO_SQL_TYPE_MAP.items():
        if sql_type_value == sql_type:
            return vtl_type
    return "String"  # Default to String if no match found


def sql_column_type_promotion(relation: Any, dtype: Any, join_key: str):
    get_type_promotion_query = f"""
        SELECT
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM {relation.alias}
                    WHERE NOT REGEXP_MATCHES({relation.alias}.{join_key}, '^[0-9]+(\\.[0-9]+)?$')
                ) THEN 'VARCHAR'
                WHEN EXISTS (
                    SELECT 1
                    FROM {relation.alias}
                    WHERE REGEXP_MATCHES({relation.alias}.{join_key}, '^[0-9]+\\.[0-9]+$')
                ) THEN 'FLOAT'
                ELSE 'INTEGER'
            END AS promoted_type
        FROM
            {relation.alias}
        LIMIT 1;
    """

    dtype = con.query(get_type_promotion_query).fetchone()
    dtype = dtype[0] if dtype else "VARCHAR"

    promotion_query = f"""
        WITH promotion_keys AS (
            SELECT
                CAST({relation.alias}.{join_key} AS {dtype}) AS {join_key}, 
                {', '.join([col for col in relation.columns if col != join_key])}
            FROM
                {relation.alias}
        )
        SELECT * FROM promotion_keys AS {relation.alias};
    """

    return con.query(promotion_query)
