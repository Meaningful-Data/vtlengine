-- ============================================================================
-- VTL Time Operator Macros for DuckDB
-- ============================================================================
-- Per-operator SQL macros for time operators in the DuckDB transpiler.
-- Depends on types and macros defined in init.sql (vtl_time_period,
-- vtl_period_parse, vtl_period_to_string).
--
-- Loaded after init.sql by initialize_time_types().
-- ============================================================================


-- ============================================================================
-- SHARED HELPERS
-- ============================================================================

-- Period limit per indicator (max periods per year)
CREATE OR REPLACE MACRO vtl_period_limit(indicator VARCHAR) AS (
    CASE indicator
        WHEN 'A' THEN 1 WHEN 'S' THEN 2 WHEN 'Q' THEN 4
        WHEN 'M' THEN 12 WHEN 'W' THEN 52 WHEN 'D' THEN 365
    END
);

-- TimePeriod → end DATE
CREATE OR REPLACE MACRO vtl_tp_end_date(p vtl_time_period) AS (
    CASE p.period_indicator
        WHEN 'A' THEN MAKE_DATE(p.year, 12, 31)
        WHEN 'S' THEN MAKE_DATE(p.year, p.period_number * 6,
            CASE p.period_number WHEN 1 THEN 30 ELSE 31 END)
        WHEN 'Q' THEN LAST_DAY(MAKE_DATE(p.year, p.period_number * 3, 1))
        WHEN 'M' THEN LAST_DAY(MAKE_DATE(p.year, p.period_number, 1))
        WHEN 'W' THEN CAST(STRPTIME(
            CAST(p.year AS VARCHAR) || '-W'
            || LPAD(CAST(p.period_number AS VARCHAR), 2, '0') || '-7',
            '%G-W%V-%u') AS DATE)
        WHEN 'D' THEN CAST(MAKE_DATE(p.year, 1, 1)
            + INTERVAL (p.period_number - 1) DAY AS DATE)
    END
);

-- TimePeriod → start DATE
CREATE OR REPLACE MACRO vtl_tp_start_date(p vtl_time_period) AS (
    CASE p.period_indicator
        WHEN 'A' THEN MAKE_DATE(p.year, 1, 1)
        WHEN 'S' THEN MAKE_DATE(p.year, (p.period_number - 1) * 6 + 1, 1)
        WHEN 'Q' THEN MAKE_DATE(p.year, (p.period_number - 1) * 3 + 1, 1)
        WHEN 'M' THEN MAKE_DATE(p.year, p.period_number, 1)
        WHEN 'W' THEN CAST(STRPTIME(
            CAST(p.year AS VARCHAR) || '-W'
            || LPAD(CAST(p.period_number AS VARCHAR), 2, '0') || '-1',
            '%G-W%V-%u') AS DATE)
        WHEN 'D' THEN CAST(MAKE_DATE(p.year, 1, 1)
            + INTERVAL (p.period_number - 1) DAY AS DATE)
    END
);


-- ============================================================================
-- OPERATOR: getmonth (TimePeriod → INTEGER)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_tp_getmonth(p vtl_time_period) AS (
    CASE p.period_indicator
        WHEN 'A' THEN 1
        WHEN 'S' THEN (p.period_number - 1) * 6 + 1
        WHEN 'Q' THEN (p.period_number - 1) * 3 + 1
        WHEN 'M' THEN p.period_number
        WHEN 'W' THEN MONTH(CAST(STRPTIME(
            CAST(p.year AS VARCHAR) || '-W'
            || LPAD(CAST(p.period_number AS VARCHAR), 2, '0') || '-1',
            '%G-W%V-%u') AS DATE))
        WHEN 'D' THEN MONTH(CAST(MAKE_DATE(p.year, 1, 1)
            + INTERVAL (p.period_number - 1) DAY AS DATE))
    END
);


-- ============================================================================
-- OPERATOR: dayofmonth (TimePeriod → INTEGER)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_tp_dayofmonth(p vtl_time_period) AS (
    DAY(vtl_tp_end_date(p))
);


-- ============================================================================
-- OPERATOR: dayofyear (TimePeriod → INTEGER)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_tp_dayofyear(p vtl_time_period) AS (
    CASE p.period_indicator
        WHEN 'D' THEN p.period_number
        ELSE DAYOFYEAR(vtl_tp_end_date(p))
    END
);


-- ============================================================================
-- OPERATOR: datediff (TimePeriod × TimePeriod → INTEGER)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_tp_datediff(a vtl_time_period, b vtl_time_period) AS (
    ABS(DATE_DIFF('day', vtl_tp_end_date(a), vtl_tp_end_date(b)))
);


-- ============================================================================
-- OPERATOR: dateadd (Date/TimePeriod + shift + period → Date)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_dateadd(d, shift INTEGER, period_ind VARCHAR) AS (
    CASE period_ind
        WHEN 'D' THEN CAST(d AS TIMESTAMP) + INTERVAL (shift) DAY
        WHEN 'W' THEN CAST(d AS TIMESTAMP) + INTERVAL (shift * 7) DAY
        WHEN 'M' THEN CAST(d AS TIMESTAMP) + INTERVAL (shift) MONTH
        WHEN 'Q' THEN CAST(d AS TIMESTAMP) + INTERVAL (shift * 3) MONTH
        WHEN 'S' THEN CAST(d AS TIMESTAMP) + INTERVAL (shift * 6) MONTH
        WHEN 'A' THEN CAST(d AS TIMESTAMP) + INTERVAL (shift) YEAR
    END
);

CREATE OR REPLACE MACRO vtl_tp_dateadd(
    p vtl_time_period, shift INTEGER, period_ind VARCHAR
) AS (
    vtl_dateadd(vtl_tp_end_date(p), shift, period_ind)
);

-- Duration mapping

CREATE OR REPLACE MACRO vtl_duration_to_int(d) AS (
    CASE d
        WHEN 'A' THEN 6
        WHEN 'S' THEN 5
        WHEN 'Q' THEN 4
        WHEN 'M' THEN 3
        WHEN 'W' THEN 2
        WHEN 'D' THEN 1
        ELSE NULL
    END
);

CREATE OR REPLACE MACRO vtl_int_to_duration(i) AS (
    CASE i
        WHEN 6 THEN 'A'
        WHEN 5 THEN 'S'
        WHEN 4 THEN 'Q'
        WHEN 3 THEN 'M'
        WHEN 2 THEN 'W'
        WHEN 1 THEN 'D'
        ELSE NULL
    END
);


-- ============================================================================
-- OPERATOR: daytoyear / daytomonth (Integer → Duration VARCHAR)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_daytoyear(days) AS (
        'P' || CAST(days // 365 AS VARCHAR) || 'Y'
        || CAST(days % 365 AS VARCHAR) || 'D'
);

CREATE OR REPLACE MACRO vtl_daytomonth(days) AS (
        'P' || CAST(days // 30 AS VARCHAR) || 'M'
        || CAST(days % 30 AS VARCHAR) || 'D'
);


-- ============================================================================
-- OPERATOR: yeartoday / monthtoday (Duration VARCHAR → Integer)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_yeartoday(dur) AS (
    CASE WHEN dur IS NULL THEN 
        NULL 
    ELSE
        COALESCE(TRY_CAST(REGEXP_EXTRACT(dur, '(\d+)Y', 1) AS INTEGER), 0) * 365
        + COALESCE(TRY_CAST(REGEXP_EXTRACT(dur, '(\d+)D', 1) AS INTEGER), 0)
    END
);

CREATE OR REPLACE MACRO vtl_monthtoday(dur) AS (
    CASE WHEN dur IS NULL THEN 
        NULL 
    ELSE
        COALESCE(TRY_CAST(REGEXP_EXTRACT(dur, '(\d+)M', 1) AS INTEGER), 0) * 30
        + COALESCE(TRY_CAST(REGEXP_EXTRACT(dur, '(\d+)D', 1) AS INTEGER), 0)
    END
);


-- ============================================================================
-- OPERATOR: time_agg (Date/TimePeriod → TimePeriod)
-- ============================================================================

-- Date → TimePeriod internal representation
CREATE OR REPLACE MACRO vtl_time_agg_date(d, target VARCHAR) AS (
    CASE target
        WHEN 'A' THEN CAST(YEAR(d) AS VARCHAR) || 'A'
        WHEN 'S' THEN CAST(YEAR(d) AS VARCHAR) || '-S'
            || CAST(((MONTH(d) - 1) // 6) + 1 AS VARCHAR)
        WHEN 'Q' THEN CAST(YEAR(d) AS VARCHAR) || '-Q'
            || CAST(QUARTER(d) AS VARCHAR)
        WHEN 'M' THEN CAST(YEAR(d) AS VARCHAR) || '-M'
            || LPAD(CAST(MONTH(d) AS VARCHAR), 2, '0')
        WHEN 'W' THEN CAST(ISOYEAR(d) AS VARCHAR) || '-W'
            || LPAD(CAST(WEEK(d) AS VARCHAR), 2, '0')
        WHEN 'D' THEN CAST(YEAR(d) AS VARCHAR) || '-D'
            || LPAD(CAST(DAYOFYEAR(d) AS VARCHAR), 3, '0')
    END
);

-- TimePeriod → TimePeriod (convert via end_date)
CREATE OR REPLACE MACRO vtl_time_agg_tp(p vtl_time_period, target VARCHAR) AS (
    CASE
        WHEN p.period_indicator = target THEN vtl_period_to_string(p)
        ELSE vtl_time_agg_date(vtl_tp_end_date(p), target)
    END
);


-- ============================================================================
-- OPERATOR: timeshift (TimePeriod shift by N periods)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_tp_shift(p vtl_time_period, n INTEGER) AS (
    CASE p.period_indicator
        WHEN 'A' THEN
            vtl_period_to_string({'year': p.year + n,
                'period_indicator': 'A', 'period_number': 1}::vtl_time_period)
        ELSE
            vtl_period_to_string({
                'year': p.year + CASE
                    WHEN p.period_number + n <= 0 THEN
                        (p.period_number + n) // vtl_period_limit(p.period_indicator) - 1
                    ELSE
                        (p.period_number + n - 1) // vtl_period_limit(p.period_indicator)
                END,
                'period_indicator': p.period_indicator,
                'period_number':
                    ((p.period_number + n - 1)
                        % vtl_period_limit(p.period_indicator)
                        + vtl_period_limit(p.period_indicator))
                    % vtl_period_limit(p.period_indicator) + 1
            }::vtl_time_period)
    END
);
