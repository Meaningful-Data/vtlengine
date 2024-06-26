from AST.Grammar.tokens import *
from Operators.General import Membership
from Operators.Comparison import Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual
from Operators.Numeric import UnPlus, UnMinus,AbsoluteValue, Exponential, NaturalLogarithm, SquareRoot, BinPlus, BinMinus, Mult, Div, Logarithm

from Operators.RegularAggregation import Calc
from Operators.RoleSetter import Identifier, Attribute, Measure
from Operators.String import Length, Concatenate

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
    # IN: In,
    # Numeric
    PLUS: BinPlus,
    MINUS: BinMinus,
    MULT: Mult,
    DIV: Div,
    LOG: Logarithm,
    # String
    CONCAT: Concatenate
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
    # Role Setter
    IDENTIFIER: Identifier,
    ATTRIBUTE: Attribute,
    MEASURE: Measure,
    # String
    LEN: Length
}

REGULAR_AGGREGATION_MAPPING = {
    CALC: Calc
}
