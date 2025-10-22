from typing import Dict, Union, Any

from vtlengine.AST.Grammar.tokens import (
    CHARSET_MATCH,
    DATEDIFF,
    DAYOFMONTH,
    DAYTOMONTH,
    DAYTOYEAR,
    DIV,
    ISNULL,
    LOG,
    MOD,
    MONTHTODAY,
    NOT_IN,
    POWER,
    XOR,
    YEARTODAY,
)

# Could it be the operator sql token or a tuple of (sql token, token position)
# default is taken as MIDDLE on Operator apply_operation method
MIDDLE = "middle"
LEFT = "left"

# If an operator must be on the left side of the expression on bin operators
# return a tuple with the sql token and the position, whenever the position is
# not specified it is assumed to be MIDDLE
TO_SQL_TOKEN: Dict[str, Union[str, tuple[str, str]]] = {
    # Numeric operators
    DIV: ("division_duck", LEFT),
    MOD: "%",
    POWER: "^",
    LOG: (LOG, LEFT),
    XOR: (XOR, LEFT),
    CHARSET_MATCH: ("REGEXP_MATCHES", LEFT),
    NOT_IN: "NOT IN",
    ISNULL: ("isnull_duck", LEFT),
    # Time operators
    DAYOFMONTH: ("day_of_month_duck", LEFT),
    DAYTOMONTH: ("day_to_month_duck", LEFT),
    DAYTOYEAR: ("day_to_year_duck", LEFT),
    MONTHTODAY: ("month_to_day_duck", LEFT),
    YEARTODAY: ("year_to_day_duck", LEFT),
    DATEDIFF: ("date_diff_duck", LEFT),
}


def to_sql_literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return "'" + str(v).replace("'", "''") + "'"
