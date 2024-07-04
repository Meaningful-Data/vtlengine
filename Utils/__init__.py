from Operators.Aggregation import (Avg, Count, Max, Median)
from Operators.Boolean import And, Not, Or, Xor

from Operators.Comparison import In, IsNull

Min,
PopulationStandardDeviation,
PopulationVariance, SampleStandardDeviation, SampleVariance, Sum)
from Operators.Analytic import (Avg as AvgAnalytic, Count as CountAnalytic, FirstValue, Lag,
                                LastValue, Lead, Max as MaxAnalytic, Median as MedianAnalytic,
                                Min as MinAnalytic,
                                PopulationStandardDeviation as PopulationStandardDeviationAnalytic,
                                PopulationVariance as PopulationVarianceAnalytic, Rank,
                                RatioToReport,
                                SampleStandardDeviation as SampleStandardDeviationAnalytic,
                                SampleVariance as SampleVarianceAnalytic, Sum as SumAnalytic)
from Operators.Clause import Aggregate, Drop, Filter, Keep, Pivot, Rename, Sub, Unpivot

from AST.Grammar.tokens import *
from Operators.Comparison import Equal, Greater, GreaterEqual, Less, LessEqual, NotEqual
from Operators.General import Alias, Membership
from Operators.Join import CrossJoin, FullJoin, InnerJoin, LeftJoin
from Operators.RegularAggregation import Calc
from Operators.String import Length, Concatenate, Upper, Lower, Rtrim, Ltrim, Trim, Substr, Replace
from Operators.Numeric import AbsoluteValue, BinMinus, BinPlus, Ceil, Div, Exponential, Floor, \
    Logarithm, Modulo, Mult, NaturalLogarithm, Power, SquareRoot, UnMinus, UnPlus, Trunc, Round
from Operators.RoleSetter import Attribute, Identifier, Measure
from Operators.Set import Intersection, Setdiff, Symdiff, Union

BINARY_MAPPING = {
    # General
    MEMBERSHIP: Membership,
    # Boolean
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
    LOG: Logarithm,
    MOD: Modulo,
    POWER: Power,
    DIV: Div,
    # General
    AS: Alias,
    # String
    CONCAT: Concatenate
}

UNARY_MAPPING = {
    # Boolean
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
    # String
    LEN: Length,
    LCASE: Lower,
    LTRIM: Ltrim,
    RTRIM: Rtrim,
    TRIM: Trim,
    UCASE: Upper
}

PARAMETRIC_MAPPING = {
    # Numeric
    ROUND: Round,
    TRUNC: Trunc,
    # String
    SUBSTR: Substr,
    REPLACE: Replace
}

ROLE_SETTER_MAPPING = {
    IDENTIFIER: Identifier,
    ATTRIBUTE: Attribute,
    MEASURE: Measure,
}

REGULAR_AGGREGATION_MAPPING = {
    CALC: Calc,
    FILTER: Filter,
    KEEP: Keep,
    DROP: Drop,
    RENAME: Rename,
    PIVOT: Pivot,
    UNPIVOT: Unpivot,
    SUBSPACE: Sub,
    AGGREGATE: Aggregate
}

SET_MAPPING = {
    UNION: Union,
    INTERSECT: Intersection,
    SYMDIFF: Symdiff,
    SETDIFF: Setdiff
}

AGGREGATION_MAPPING = {
    MAX: Max,
    MIN: Min,
    SUM: Sum,
    COUNT: Count,
    AVG: Avg,
    MEDIAN: Median,
    STDDEV_POP: PopulationStandardDeviation,
    STDDEV_SAMP: SampleStandardDeviation,
    VAR_POP: PopulationVariance,
    VAR_SAMP: SampleVariance,

}

ANALYTIC_MAPPING = {
    MAX: MaxAnalytic,
    MIN: MinAnalytic,
    SUM: SumAnalytic,
    COUNT: CountAnalytic,
    AVG: AvgAnalytic,
    MEDIAN: MedianAnalytic,
    STDDEV_POP: PopulationStandardDeviationAnalytic,
    STDDEV_SAMP: SampleStandardDeviationAnalytic,
    VAR_POP: PopulationVarianceAnalytic,
    VAR_SAMP: SampleVarianceAnalytic,
    LAG: Lag,
    LEAD: Lead,
    FIRST_VALUE: FirstValue,
    LAST_VALUE: LastValue,
    RATIO_TO_REPORT: RatioToReport,
    RANK: Rank
}

JOIN_MAPPING = {
    INNER_JOIN: InnerJoin,
    LEFT_JOIN: LeftJoin,
    FULL_JOIN: FullJoin,
    CROSS_JOIN: CrossJoin
}
