import inspect
import uuid

import duckdb


def unique_view_alias():
    return f"rel_{uuid.uuid4().hex}"


def sql_value_parser(value):
    if isinstance(value, list):
        return sql_value_parser(value[0])
    if value is None or (isinstance(value, float) and value != value):
        return "NULL"
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return repr(value)


def register_dynamic_function(func):
    if not callable(func):
        raise ValueError("`func` must be a callable function.")

    # Generate a unique name for the UDF
    func_name = f"udf_{uuid.uuid4().hex}"

    # Inspect the function signature
    sig = inspect.signature(func)
    param_count = len(sig.parameters)

    # Define input types dynamically (default to FLOAT for simplicity)
    input_types = [duckdb.FLOAT] * param_count
    return_type = duckdb.FLOAT  # Default return type, can be adjusted

    # Register the function in DuckDB
    duckdb.create_function(func_name, func, input_types, return_type)

    return func_name


def _all(self, axis=0):
    if axis != 0:
        raise NotImplementedError(f"All axis {axis} is not implemented")

    alias = unique_view_alias()
    cols = [
        f"SUM(({col} IS NOT NULL AND {col} != 0)::INT) = COUNT({col}) AS {col}"
        for col in self.columns
    ]
    query = f"SELECT {', '.join(cols)} FROM {alias}"
    result = self.query(alias, query).fetchdf().astype(bool)
    return result.values.all()


def _any(self, axis=0):
    if axis != 0:
        raise NotImplementedError("Any axis != 0 is not implemented")

    alias = unique_view_alias()
    cols = [f"SUM(({col} IS NOT NULL AND {col} != 0)::INT) > 0 AS {col}" for col in self.columns]
    query = f"SELECT {', '.join(cols)} FROM {alias}"
    result = self.query(alias, query).fetchdf().astype(bool)
    return result.values.any()


def _fillna(self, value):
    alias = unique_view_alias()
    sql_value = sql_value_parser(value)
    cols = [f"COALESCE({col}, {sql_value}) AS {col}" for col in self.columns]
    query = f"SELECT {', '.join(cols)} FROM {alias}"
    result = self.query(alias, query)
    return result


def _isnull(self):
    alias = unique_view_alias()
    cols = [f"({col} IS NULL)::INT AS {col}" for col in self.columns]
    query = f"SELECT {', '.join(cols)} FROM {alias}"
    result = self.query(alias, query)
    return result


def _map(self, func, na_action="ignore"):
    if not callable(func):
        raise ValueError("`func` must be a callable function.")

    get_lambda_name = lambda l: inspect.getsource(l).split('=')[0].strip()
    func_name = get_lambda_name(func)

    alias = unique_view_alias()
    func_name = register_dynamic_function(func)

    cols_query = [
        f"CASE WHEN {col} IS NOT NULL THEN {func_name}({col}) ELSE {col} END AS {col}"
        if na_action == "ignore" else f"{func_name}({col}) AS {col}"
        for col in self.columns
    ]

    query = f"SELECT {', '.join(cols_query)} FROM {alias}"
    return self.query(alias, query)


def _replace(self, to_replace, value=None):
    if value is None:
        raise ValueError("`value` must be specified when using `replace`.")

    alias = unique_view_alias()
    sql_to_replace = sql_value_parser(to_replace)
    sql_value = sql_value_parser(value)
    cols = [
        f"CASE WHEN {col} = {sql_to_replace} THEN {sql_value} ELSE {col} END AS {col}"
        for col in self.columns
    ]
    query = f"SELECT {', '.join(cols)} FROM {alias}"
    result = self.query(alias, query)
    return result


def set_attributes():
    duckdb.DuckDBPyRelation.all = _all
    duckdb.DuckDBPyRelation.any = _any
    duckdb.DuckDBPyRelation.fillna = _fillna
    duckdb.DuckDBPyRelation.isna = _isnull
    duckdb.DuckDBPyRelation.isnull = _isnull
    duckdb.DuckDBPyRelation.map = _map
    duckdb.DuckDBPyRelation.replace = _replace
