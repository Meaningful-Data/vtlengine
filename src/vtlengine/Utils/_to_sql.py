from typing import Union, Dict

from vtlengine.AST.Grammar.tokens import MOD, POWER

# Could be the operator sql token or a tuple of (sql token, token position)
# default is taken as MIDDLE on Operator apply_operation method
MIDDLE = "middle"
LEFT = "left"

TO_SQL_TOKEN: Dict[str, Union[str, tuple[str, str]]] = {
    # Numeric operators
    MOD: "%",
    POWER: "^",
}