from typing import Any, Dict

from pysdmx.model.dataflow import Role

from vtlengine.AST.Grammar.tokens import (
    ABS,
    AGGREGATE,
    AND,
    APPLY,
    AS,
    ATTRIBUTE,
    AVG,
    CALC,
    CEIL,
    CHARSET_MATCH,
    CONCAT,
    COUNT,
    CROSS_JOIN,
    DATE_ADD,
    DATEDIFF,
    DAYOFMONTH,
    DAYOFYEAR,
    DAYTOMONTH,
    DAYTOYEAR,
    DIV,
    DROP,
    EQ,
    EXP,
    FILL_TIME_SERIES,
    FILTER,
    FIRST_VALUE,
    FLOOR,
    FLOW_TO_STOCK,
    FULL_JOIN,
    GT,
    GTE,
    IDENTIFIER,
    IN,
    INNER_JOIN,
    INTERSECT,
    ISNULL,
    KEEP,
    LAG,
    LAST_VALUE,
    LCASE,
    LEAD,
    LEFT_JOIN,
    LEN,
    LN,
    LOG,
    LT,
    LTE,
    LTRIM,
    MAX,
    MEASURE,
    MEDIAN,
    MEMBERSHIP,
    MIN,
    MINUS,
    MOD,
    MONTH,
    MONTHTODAY,
    MULT,
    NEQ,
    NOT,
    NOT_IN,
    NVL,
    OR,
    PERIOD_INDICATOR,
    PIVOT,
    PLUS,
    POWER,
    RANDOM,
    RANK,
    RATIO_TO_REPORT,
    RENAME,
    REPLACE,
    ROUND,
    RTRIM,
    SETDIFF,
    SQRT,
    STDDEV_POP,
    STDDEV_SAMP,
    STOCK_TO_FLOW,
    SUBSPACE,
    SUBSTR,
    SUM,
    SYMDIFF,
    TIMESHIFT,
    TRIM,
    TRUNC,
    UCASE,
    UNION,
    UNPIVOT,
    VAR_POP,
    VAR_SAMP,
    XOR,
    YEAR,
    YEARTODAY,
)
from vtlengine.Operators.Aggregation import (
    Avg,
    Count,
    Max,
    Median,
    Min,
    PopulationStandardDeviation,
    PopulationVariance,
    SampleStandardDeviation,
    SampleVariance,
    Sum,
)
from vtlengine.Operators.Analytic import (
    Avg as AvgAnalytic,
)
from vtlengine.Operators.Analytic import (
    Count as CountAnalytic,
)
from vtlengine.Operators.Analytic import (
    FirstValue,
    Lag,
    LastValue,
    Lead,
    Rank,
    RatioToReport,
)
from vtlengine.Operators.Analytic import (
    Max as MaxAnalytic,
)
from vtlengine.Operators.Analytic import (
    Median as MedianAnalytic,
)
from vtlengine.Operators.Analytic import (
    Min as MinAnalytic,
)
from vtlengine.Operators.Analytic import (
    PopulationStandardDeviation as PopulationStandardDeviationAnalytic,
)
from vtlengine.Operators.Analytic import (
    PopulationVariance as PopulationVarianceAnalytic,
)
from vtlengine.Operators.Analytic import (
    SampleStandardDeviation as SampleStandardDeviationAnalytic,
)
from vtlengine.Operators.Analytic import (
    SampleVariance as SampleVarianceAnalytic,
)
from vtlengine.Operators.Analytic import (
    Sum as SumAnalytic,
)
from vtlengine.Operators.Boolean import And, Not, Or, Xor
from vtlengine.Operators.Clause import (
    Aggregate,
    Calc,
    Drop,
    Filter,
    Keep,
    Pivot,
    Rename,
    Sub,
    Unpivot,
)
from vtlengine.Operators.Comparison import (
    Equal,
    Greater,
    GreaterEqual,
    In,
    IsNull,
    Less,
    LessEqual,
    Match,
    NotEqual,
    NotIn,
)
from vtlengine.Operators.Conditional import Nvl
from vtlengine.Operators.General import Alias, Membership
from vtlengine.Operators.HROperators import (
    HRBinMinus,
    HRBinPlus,
    HREqual,
    HRGreater,
    HRGreaterEqual,
    HRLess,
    HRLessEqual,
    HRUnMinus,
    HRUnPlus,
)
from vtlengine.Operators.Join import Apply, CrossJoin, FullJoin, InnerJoin, LeftJoin
from vtlengine.Operators.Numeric import (
    AbsoluteValue,
    BinMinus,
    BinPlus,
    Ceil,
    Div,
    Exponential,
    Floor,
    Logarithm,
    Modulo,
    Mult,
    NaturalLogarithm,
    Power,
    Random,
    Round,
    SquareRoot,
    Trunc,
    UnMinus,
    UnPlus,
)
from vtlengine.Operators.RoleSetter import Attribute, Identifier, Measure
from vtlengine.Operators.Set import Intersection, Setdiff, Symdiff, Union
from vtlengine.Operators.String import (
    Concatenate,
    Length,
    Lower,
    Ltrim,
    Replace,
    Rtrim,
    Substr,
    Trim,
    Upper,
)
from vtlengine.Operators.Time import (
    Date_Add,
    Date_Diff,
    Day_of_Month,
    Day_of_Year,
    Day_to_Month,
    Day_to_Year,
    Fill_time_series,
    Flow_to_stock,
    Month,
    Month_to_Day,
    Period_indicator,
    Stock_to_flow,
    Time_Shift,
    Year,
    Year_to_Day,
)

BINARY_MAPPING: Dict[Any, Any] = {
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
    NOT_IN: NotIn,
    # Conditional
    NVL: Nvl,
    # Numeric
    PLUS: BinPlus,
    MINUS: BinMinus,
    MULT: Mult,
    LOG: Logarithm,
    MOD: Modulo,
    POWER: Power,
    DIV: Div,
    RANDOM: Random,
    # General
    AS: Alias,
    # String
    CONCAT: Concatenate,
    # Time
    TIMESHIFT: Time_Shift,
    CHARSET_MATCH: Match,
    DATEDIFF: Date_Diff,
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
    UCASE: Upper,
    # Time
    PERIOD_INDICATOR: Period_indicator,
    FLOW_TO_STOCK: Flow_to_stock,
    STOCK_TO_FLOW: Stock_to_flow,
    YEAR: Year,
    MONTH: Month,
    DAYOFMONTH: Day_of_Month,
    DAYOFYEAR: Day_of_Year,
    DAYTOYEAR: Day_to_Year,
    DAYTOMONTH: Day_to_Month,
    YEARTODAY: Year_to_Day,
    MONTHTODAY: Month_to_Day,
}

PARAMETRIC_MAPPING = {
    # Numeric
    ROUND: Round,
    TRUNC: Trunc,
    # String
    SUBSTR: Substr,
    REPLACE: Replace,
    # Time
    FILL_TIME_SERIES: Fill_time_series,
    DATE_ADD: Date_Add,
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
    AGGREGATE: Aggregate,
    APPLY: Apply,
}

SET_MAPPING = {
    UNION: Union,
    INTERSECT: Intersection,
    SYMDIFF: Symdiff,
    SETDIFF: Setdiff,
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
    RANK: Rank,
}

THEN_ELSE = {"then": "T", "else": "E"}

JOIN_MAPPING = {
    INNER_JOIN: InnerJoin,
    LEFT_JOIN: LeftJoin,
    FULL_JOIN: FullJoin,
    CROSS_JOIN: CrossJoin,
}

HR_COMP_MAPPING = {
    # Comparison
    EQ: HREqual,
    GT: HRGreater,
    GTE: HRGreaterEqual,
    LT: HRLess,
    LTE: HRLessEqual,
}

HR_NUM_BINARY_MAPPING = {
    # Numeric
    PLUS: HRBinPlus,
    MINUS: HRBinMinus,
}

HR_UNARY_MAPPING = {
    # Numeric
    PLUS: HRUnPlus,
    MINUS: HRUnMinus,
}

HA_COMP_MAPPING = {
    # Comparison
    EQ: HREqual,
    GT: HRGreater,
    GTE: HRGreaterEqual,
    LT: HRLess,
    LTE: HRLessEqual,
}

HA_NUM_BINARY_MAPPING = {
    # Numeric
    PLUS: HRBinPlus,
    MINUS: HRBinMinus,
}

HA_UNARY_MAPPING = {
    # Numeric
    PLUS: HRUnPlus,
    MINUS: HRUnMinus,
}
VTL_DTYPES_MAPPING = {
    "String": "String",
    "Alpha": "String",
    "AlphaNumeric": "String",
    "Numeric": "String",
    "BigInteger": "Integer",
    "Integer": "Integer",
    "Long": "Integer",
    "Short": "Integer",
    "Decimal": "Number",
    "Float": "Number",
    "Double": "Number",
    "Boolean": "Boolean",
    "URI": "String",
    "Count": "Integer",
    "InclusiveValueRange": "Number",
    "ExclusiveValueRange": "Number",
    "Incremental": "Number",
    "ObservationalTimePeriod": "Time_Period",
    "StandardTimePeriod": "Time_Period",
    "BasicTimePeriod": "Date",
    "GregorianTimePeriod": "Date",
    "GregorianYear": "Date",
    "GregorianYearMonth": "Date",
    "GregorianMonth": "Date",
    "GregorianDay": "Date",
    "ReportingTimePeriod": "Time_Period",
    "ReportingYear": "Time_Period",
    "ReportingSemester": "Time_Period",
    "ReportingTrimester": "Time_Period",
    "ReportingQuarter": "Time_Period",
    "ReportingMonth": "Time_Period",
    "ReportingWeek": "Time_Period",
    "ReportingDay": "Time_Period",
    "DateTime": "Date",
    "TimeRange": "Time",
    "Month": "String",
    "MonthDay": "String",
    "Day": "String",
    "Time": "String",
    "Duration": "Duration",
}
VTL_ROLE_MAPPING = {
    Role.DIMENSION: "Identifier",
    Role.MEASURE: "Measure",
    Role.ATTRIBUTE: "Attribute",
}
