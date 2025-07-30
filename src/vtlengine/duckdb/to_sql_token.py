from typing import Dict, Union

from vtlengine.AST.Grammar.tokens import CHARSET_MATCH, ISNULL, LOG, MOD, NOT_IN, POWER, XOR

# Could it be the operator sql token or a tuple of (sql token, token position)
# default is taken as MIDDLE on Operator apply_operation method
MIDDLE = "middle"
LEFT = "left"

# If an operator must be on the left side of the expression on bin operators
# return a tuple with the sql token and the position, whenever the position is
# not specified it is assumed to be MIDDLE
TO_SQL_TOKEN: Dict[str, Union[str, tuple[str, str]]] = {
    # Numeric operators
    MOD: "%",
    POWER: "^",
    LOG: (LOG, LEFT),
    XOR: (XOR, LEFT),
    CHARSET_MATCH: ("REGEXP_MATCHES", LEFT),
    NOT_IN: "NOT IN",
    ISNULL: "isnull_duck",
}
