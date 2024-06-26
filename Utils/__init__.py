from AST.Grammar.tokens import *
from Operators.General import Alias, Membership
from Operators.Comparison import Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual
from Operators.Join import CrossJoin, FullJoin, InnerJoin, LeftJoin
from Operators.Numeric import UnPlus, UnMinus,AbsoluteValue, Exponential, NaturalLogarithm, SquareRoot, BinPlus, BinMinus, Mult, Div, Logarithm

from Operators.RegularAggregation import Calc
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
    # IN: In,
    # Numeric
    PLUS: BinPlus,
    MINUS: BinMinus,
    MULT: Mult,
    DIV: Div,
    LOG: Logarithm,
    # General
    AS: Alias
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
    MEASURE: Measure
}

REGULAR_AGGREGATION_MAPPING = {
    CALC: Calc
}

JOIN_MAPPING = {
    INNER_JOIN: InnerJoin,
    LEFT_JOIN: LeftJoin,
    FULL_JOIN: FullJoin,
    CROSS_JOIN: CrossJoin
}