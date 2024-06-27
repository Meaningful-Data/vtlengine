from AST.Grammar.tokens import *
from Operators.General import Membership
from Operators.Comparison import Equal, In, IsNull, NotEqual, Greater, GreaterEqual, Less, LessEqual
from Operators.Numeric import Ceil, Floor, Modulo, Power, UnPlus, UnMinus, AbsoluteValue, Exponential, \
    NaturalLogarithm, \
    SquareRoot, BinPlus, BinMinus, Mult, Div, Logarithm

from Operators.Clause import Calc, Drop, Filter, Keep, Pivot, Rename, Sub, Unpivot
from Operators.RoleSetter import Identifier, Attribute, Measure

BINARY_MAPPING = {
    # General
    MEMBERSHIP: Membership,
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
    # Role Setter
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
