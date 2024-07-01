from Operators.Clause import Calc, Drop, Filter, Keep, Pivot, Rename, Sub, Unpivot

from AST.Grammar.tokens import *
from Operators.Boolean import Not, And, Or, Xor
from Operators.Comparison import Equal, Greater, GreaterEqual, In, IsNull, Less, LessEqual, NotEqual
from Operators.General import Membership
from Operators.Numeric import AbsoluteValue, BinMinus, BinPlus, Ceil, Div, Exponential, Floor, \
    Logarithm, Modulo, Mult, NaturalLogarithm, Power, SquareRoot, UnMinus, UnPlus
from Operators.RoleSetter import Attribute, Identifier, Measure
from Operators.Set import Intersection, Setdiff, Symdiff, Union

BINARY_MAPPING = {
    # General
    MEMBERSHIP: Membership,
    #Boolean
    AND: And,
    OR: Or,
    XOR: Xor,
    # Comparison
    EQ: Equal,
    NEQ: NotEqual,
    GT: Greater,
    GTE: GreaterEqual,
    LT: Less,
    LTE: LessEqual,
    IN: In,
    # Numeric
    PLUS: BinPlus,
    MINUS: BinMinus,
    MULT: Mult,
    DIV: Div,
    LOG: Logarithm,
    MOD: Modulo,
    POWER: Power
}

UNARY_MAPPING = {
    #Boolean
    NOT: Not,
    # Comparison
    # ISNULL: IsNull,
    # Numeric
    PLUS: UnPlus,
    MINUS: UnMinus,
    ABS: AbsoluteValue,
    EXP: Exponential,
    LN: NaturalLogarithm,
    SQRT: SquareRoot,
    CEIL: Ceil,
    FLOOR: Floor,
    ISNULL: IsNull,
}

ROLE_SETTER_MAPPING = {
    IDENTIFIER: Identifier,
    ATTRIBUTE: Attribute,
    MEASURE: Measure
}

REGULAR_AGGREGATION_MAPPING = {
    CALC: Calc,
    FILTER: Filter,
    KEEP: Keep,
    DROP: Drop,
    RENAME: Rename,
    PIVOT: Pivot,
    UNPIVOT: Unpivot,
    SUBSPACE: Sub
}

SET_MAPPING = {
    UNION: Union,
    INTERSECT: Intersection,
    SYMDIFF: Symdiff,
    SETDIFF: Setdiff
}
