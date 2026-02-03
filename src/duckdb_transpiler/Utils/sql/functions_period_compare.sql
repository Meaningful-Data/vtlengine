-- TimePeriod Comparison Functions
-- All comparison functions validate that both operands have the same period_indicator

-- Helper macro to validate same indicator
CREATE OR REPLACE MACRO vtl_period_check_same_indicator(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN TRUE
        WHEN a.period_indicator != b.period_indicator THEN
            error('VTL Error: Cannot compare TimePeriods with different indicators: ' ||
                  a.period_indicator || ' vs ' || b.period_indicator ||
                  '. Periods must have the same period indicator for comparison.')
        ELSE TRUE
    END
);

-- Less than
CREATE OR REPLACE MACRO vtl_period_lt(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        WHEN NOT vtl_period_check_same_indicator(a, b) THEN NULL
        ELSE a.start_date < b.start_date
    END
);

-- Less than or equal
CREATE OR REPLACE MACRO vtl_period_le(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        WHEN NOT vtl_period_check_same_indicator(a, b) THEN NULL
        ELSE a.start_date <= b.start_date
    END
);

-- Greater than
CREATE OR REPLACE MACRO vtl_period_gt(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        WHEN NOT vtl_period_check_same_indicator(a, b) THEN NULL
        ELSE a.start_date > b.start_date
    END
);

-- Greater than or equal
CREATE OR REPLACE MACRO vtl_period_ge(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        WHEN NOT vtl_period_check_same_indicator(a, b) THEN NULL
        ELSE a.start_date >= b.start_date
    END
);

-- Equal
CREATE OR REPLACE MACRO vtl_period_eq(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        ELSE a.start_date = b.start_date AND a.end_date = b.end_date
    END
);

-- Not equal
CREATE OR REPLACE MACRO vtl_period_ne(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        ELSE a.start_date != b.start_date OR a.end_date != b.end_date
    END
);

-- Sort key for ORDER BY and aggregations (returns days since epoch)
CREATE OR REPLACE MACRO vtl_period_sort_key(p) AS (
    CASE
        WHEN p IS NULL THEN NULL
        ELSE (p.start_date - DATE '1970-01-01')::INTEGER
    END
);
