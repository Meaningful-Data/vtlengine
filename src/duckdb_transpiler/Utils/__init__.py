from typing import Any, Dict, Optional

from vtlengine.DataTypes import Boolean, Date, Integer, Number

SQL_OP_MAPPING: Dict[str, str] = {
    "mod": "%",
    "len": "LENGTH",
    "ucase": "UPPER",
    "lcase": "LOWER",
    "isnull": "IS NULL",
}


def get_sql_op(op: str) -> str:
    """Get the SQL equivalent of a given operator."""
    return SQL_OP_MAPPING.get(op, op.upper())


def get_sql_type(vtl_type: type) -> str:
    """Get the DuckDB SQL type for a VTL scalar type."""
    mapping = {
        Integer: "BIGINT",
        Number: "DOUBLE",
        Boolean: "BOOLEAN",
        Date: "DATE",
    }

    return mapping.get(vtl_type, "VARCHAR")


def get_pandas_type(vtl_type: type) -> str:
    """Get the pandas dtype for a VTL scalar type."""
    mapping = {
        Integer: "int64",
        Number: "float64",
        Boolean: "bool",
        Date: "datetime64[ns]",
    }

    return mapping.get(vtl_type, "object")


def sql_literal(value: Any, type_: Optional[str] = None) -> str:
    """Convert a value to SQL literal."""
    if value is None or type_ == "NULL_CONSTANT":
        return "NULL"
    elif type_ in ("STRING_CONSTANT", "String") or isinstance(value, str):
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
    elif type_ in ("INTEGER_CONSTANT", "Integer"):
        return str(int(value))
    elif type_ in ("FLOAT_CONSTANT", "Number"):
        return str(float(value))
    elif type_ in ("BOOLEAN_CONSTANT", "Boolean") or isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    return str(value)
