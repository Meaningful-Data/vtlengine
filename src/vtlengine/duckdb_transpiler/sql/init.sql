-- ============================================================================
-- VTL Time Types for DuckDB
-- VTL Time Types for DuckDB
-- ============================================================================
-- Types and macros for TimePeriod and TimeInterval handling.
-- Loaded once when initializing a DuckDB connection for VTL.
--
-- Architecture:
-- 1. vtl_period_normalize: VARCHAR -> VARCHAR (any input to canonical)
-- 2. vtl_period_parse / vtl_period_to_string: VARCHAR <-> vtl_time_period
-- 3. vtl_period_lt/le/gt/ge: vtl_time_period ordering with indicator check
-- 4. Equality (=, <>): native VARCHAR comparison (no macros needed)
-- 5. Representation macros: VARCHAR -> VARCHAR (canonical to output format)
-- ============================================================================


-- ============================================================================
-- TYPE DEFINITIONS
-- ============================================================================

DROP TYPE IF EXISTS vtl_time_period;
DROP TYPE IF EXISTS vtl_time_interval;

-- Mirrors TimePeriodHandler: _year, _period_indicator, _period_number
CREATE TYPE vtl_time_period AS STRUCT(
    year INTEGER,
    period_indicator VARCHAR,
    period_number INTEGER
);

-- Mirrors TimeIntervalHandler: _date1, _date2
CREATE TYPE vtl_time_interval AS STRUCT(
    date1 DATE,
    date2 DATE
);


-- ============================================================================
-- NORMALIZE: VARCHAR -> VARCHAR
-- ============================================================================
-- Any input format (#505) -> canonical internal representation.
-- Runs once at data load time. All subsequent operations use the normalized form.
-- Reference: from_input_customer_support_to_internal (TimeHandling.py:79-110)

CREATE OR REPLACE MACRO vtl_period_normalize(input VARCHAR) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN LENGTH(input) = 4 THEN
            input || 'A'
        WHEN SUBSTR(input, 5, 1) != '-' THEN
            CASE
                WHEN UPPER(SUBSTR(input, 5, 1)) = 'A' THEN
                    SUBSTR(input, 1, 4) || 'A'
                WHEN UPPER(SUBSTR(input, 5, 1)) IN ('S', 'Q') THEN
                    SUBSTR(input, 1, 4) || '-' || UPPER(SUBSTR(input, 5, 1))
                    || CAST(CAST(SUBSTR(input, 6) AS INTEGER) AS VARCHAR)
                WHEN UPPER(SUBSTR(input, 5, 1)) IN ('M', 'W') THEN
                    SUBSTR(input, 1, 4) || '-' || UPPER(SUBSTR(input, 5, 1))
                    || LPAD(CAST(CAST(SUBSTR(input, 6) AS INTEGER) AS VARCHAR), 2, '0')
                ELSE
                    SUBSTR(input, 1, 4) || '-D'
                    || LPAD(CAST(CAST(SUBSTR(input, 6) AS INTEGER) AS VARCHAR), 3, '0')
            END
        WHEN UPPER(SUBSTR(input, 6, 1)) >= 'A' AND UPPER(SUBSTR(input, 6, 1)) <= 'Z' THEN
            CASE
                WHEN UPPER(SUBSTR(input, 6, 1)) = 'A' THEN
                    SUBSTR(input, 1, 4) || 'A'
                WHEN UPPER(SUBSTR(input, 6, 1)) IN ('S', 'Q') THEN
                    SUBSTR(input, 1, 4) || '-' || UPPER(SUBSTR(input, 6, 1))
                    || CAST(TRY_CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR)
                WHEN UPPER(SUBSTR(input, 6, 1)) IN ('M', 'W') THEN
                    SUBSTR(input, 1, 4) || '-' || UPPER(SUBSTR(input, 6, 1))
                    || LPAD(CAST(TRY_CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR), 2, '0')
                ELSE
                    SUBSTR(input, 1, 4) || '-D'
                    || LPAD(CAST(TRY_CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR), 3, '0')
            END
        WHEN LENGTH(input) >= 10 AND SUBSTR(input, 5, 1) = '-'
             AND SUBSTR(input, 8, 1) = '-' THEN
            -- Full date (2020-01-15) or timestamp (2020-01-15 00:00:00) → daily period
            SUBSTR(input, 1, 4) || '-D'
            || LPAD(CAST(DAYOFYEAR(CAST(SUBSTR(input, 1, 10) AS DATE)) AS VARCHAR), 3, '0')
        ELSE
            SUBSTR(input, 1, 4) || '-M'
            || LPAD(CAST(CAST(SUBSTR(input, 6) AS INTEGER) AS VARCHAR), 2, '0')
    END
);


-- ============================================================================
-- PARSE: VARCHAR -> vtl_time_period
-- ============================================================================
-- Only handles the canonical format from TimePeriodHandler.__str__

CREATE OR REPLACE MACRO vtl_period_parse(input VARCHAR) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN SUBSTR(input, 5, 1) = '-' THEN
            {'year': CAST(SUBSTR(input, 1, 4) AS INTEGER),
             'period_indicator': SUBSTR(input, 6, 1),
             'period_number': CAST(SUBSTR(input, 7) AS INTEGER)
            }::vtl_time_period
        ELSE
            {'year': CAST(SUBSTR(input, 1, 4) AS INTEGER),
             'period_indicator': 'A',
             'period_number': 1
            }::vtl_time_period
    END
);


-- ============================================================================
-- FORMAT: vtl_time_period -> VARCHAR
-- ============================================================================
-- Reference: TimePeriodHandler.__str__ (TimeHandling.py:173-182)

CREATE OR REPLACE MACRO vtl_period_to_string(p vtl_time_period) AS (
    CASE
        WHEN p IS NULL THEN NULL
        WHEN p.period_indicator = 'A' THEN
            CAST(p.year AS VARCHAR) || 'A'
        ELSE
            CONCAT(
                CAST(p.year AS VARCHAR), '-', p.period_indicator,
                LPAD(CAST(p.period_number AS VARCHAR),
                     CASE p.period_indicator
                         WHEN 'D' THEN 3
                         WHEN 'M' THEN 2
                         WHEN 'W' THEN 2
                         ELSE 1
                     END, '0')
            )
    END
);


-- ============================================================================
-- TIMEINTERVAL PARSE/FORMAT
-- ============================================================================

CREATE OR REPLACE MACRO vtl_interval_parse(input VARCHAR) AS (
    CASE
        WHEN input IS NULL THEN NULL
        ELSE {
            'date1': CAST(SUBSTR(input, 1, 10) AS DATE),
            'date2': CAST(SUBSTR(input, 12) AS DATE)
        }::vtl_time_interval
    END
);

CREATE OR REPLACE MACRO vtl_interval_to_string(i vtl_time_interval) AS (
    CASE
        WHEN i IS NULL THEN NULL
        ELSE CAST(i.date1 AS VARCHAR) || '/' || CAST(i.date2 AS VARCHAR)
    END
);


-- ============================================================================
-- CAST MACROS: Cross-type conversions for VTL cast operator
-- ============================================================================

-- Date (TIMESTAMP) -> TimePeriod (VARCHAR): always daily period
-- Reference: date_to_period_str(value, 'D') in TimeHandling.py
CREATE OR REPLACE MACRO vtl_date_to_period(d) AS (
    CASE
        WHEN d IS NULL THEN NULL
        ELSE vtl_period_normalize(STRFTIME(CAST(d AS DATE), '%Y-%m-%d'))
    END
);

-- TimePeriod (VARCHAR) -> Date (TIMESTAMP): only daily periods allowed
-- Reference: Date.explicit_cast from TimePeriod in DataTypes/__init__.py
CREATE OR REPLACE MACRO vtl_period_to_date(tp VARCHAR) AS (
    CASE
        WHEN tp IS NULL THEN NULL
        -- Normalized daily format: 'YYYY-DXXX'
        WHEN LENGTH(tp) >= 6 AND SUBSTR(tp, 6, 1) = 'D' THEN
            CAST(MAKE_DATE(
                CAST(SUBSTR(tp, 1, 4) AS INTEGER), 1, 1
            ) + INTERVAL (CAST(SUBSTR(tp, 7) AS INTEGER) - 1) DAY AS TIMESTAMP)
        -- Non-normalized daily format: 'YYYYDXXX'
        WHEN LENGTH(tp) >= 5 AND UPPER(SUBSTR(tp, 5, 1)) = 'D' THEN
            CAST(MAKE_DATE(
                CAST(SUBSTR(tp, 1, 4) AS INTEGER), 1, 1
            ) + INTERVAL (CAST(SUBSTR(tp, 6) AS INTEGER) - 1) DAY AS TIMESTAMP)
        ELSE error('Cannot cast non-daily TimePeriod to Date: ' || tp)
    END
);

-- TimeInterval (VARCHAR) -> Date (TIMESTAMP): only same-date intervals
-- Reference: Date.explicit_cast from TimeInterval in DataTypes/__init__.py
CREATE OR REPLACE MACRO vtl_interval_to_date(interval_str VARCHAR) AS (
    CASE
        WHEN interval_str IS NULL THEN NULL
        WHEN SPLIT_PART(interval_str, '/', 1) = SPLIT_PART(interval_str, '/', 2) THEN
            CAST(SPLIT_PART(interval_str, '/', 1) AS TIMESTAMP)
        ELSE error('Cannot cast TimeInterval to Date: dates differ in ' || interval_str)
    END
);

-- TimeInterval (VARCHAR) -> TimePeriod (VARCHAR): match interval to period
-- Reference: interval_to_period_str in TimeHandling.py
-- Tries A, S, Q, M, W, D period indicators to find a match.
CREATE OR REPLACE MACRO vtl_interval_to_period(interval_str VARCHAR) AS (
    CASE
        WHEN interval_str IS NULL THEN NULL
        ELSE (SELECT CASE
            -- Day: same date
            WHEN d1 = d2 THEN
                vtl_period_normalize(CAST(d1 AS VARCHAR))
            -- Annual: Jan 1 to Dec 31
            WHEN MONTH(d1) = 1 AND DAY(d1) = 1
                 AND MONTH(d2) = 12 AND DAY(d2) = 31
                 AND YEAR(d1) = YEAR(d2)
            THEN CAST(YEAR(d1) AS VARCHAR) || 'A'
            -- Semester 1: Jan 1 to Jun 30
            WHEN MONTH(d1) = 1 AND DAY(d1) = 1
                 AND MONTH(d2) = 6 AND DAY(d2) = 30
                 AND YEAR(d1) = YEAR(d2)
            THEN CAST(YEAR(d1) AS VARCHAR) || '-S1'
            -- Semester 2: Jul 1 to Dec 31
            WHEN MONTH(d1) = 7 AND DAY(d1) = 1
                 AND MONTH(d2) = 12 AND DAY(d2) = 31
                 AND YEAR(d1) = YEAR(d2)
            THEN CAST(YEAR(d1) AS VARCHAR) || '-S2'
            -- Quarter
            WHEN DAY(d1) = 1 AND YEAR(d1) = YEAR(d2)
                 AND MONTH(d1) IN (1, 4, 7, 10)
                 AND d2 = LAST_DAY(d1 + INTERVAL 2 MONTH)
            THEN CAST(YEAR(d1) AS VARCHAR) || '-Q'
                 || CAST(((MONTH(d1) - 1) / 3 + 1) AS VARCHAR)
            -- Month
            WHEN DAY(d1) = 1 AND d2 = LAST_DAY(d1)
                 AND YEAR(d1) = YEAR(d2)
            THEN CAST(YEAR(d1) AS VARCHAR) || '-M'
                 || LPAD(CAST(MONTH(d1) AS VARCHAR), 2, '0')
            -- Week (ISO)
            WHEN ISODOW(d1) = 1 AND d2 = d1 + INTERVAL 6 DAY
            THEN CAST(ISOYEAR(d1) AS VARCHAR) || '-W'
                 || LPAD(CAST(WEEKOFYEAR(d1) AS VARCHAR), 2, '0')
            ELSE error('Cannot determine period for interval: ' || interval_str)
        END
        FROM (SELECT CAST(SPLIT_PART(interval_str, '/', 1) AS DATE) AS d1,
                     CAST(SPLIT_PART(interval_str, '/', 2) AS DATE) AS d2) AS _iv)
    END
);


-- ============================================================================
-- COMPARISON MACROS: vtl_time_period ordering (equality uses VARCHAR directly)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_period_check_indicator(
    a vtl_time_period, b vtl_time_period
) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN TRUE
        WHEN a.period_indicator != b.period_indicator THEN
            error('VTL Error 2-1-19-19: Cannot compare TimePeriods with '
                  || 'different indicators: '
                  || a.period_indicator || ' vs ' || b.period_indicator)
        ELSE TRUE
    END
);

CREATE OR REPLACE MACRO vtl_period_lt(
    a vtl_time_period, b vtl_time_period
) AS (
    CASE WHEN a IS NULL OR b IS NULL THEN NULL
         WHEN NOT vtl_period_check_indicator(a, b) THEN NULL
         ELSE a < b END
);

CREATE OR REPLACE MACRO vtl_period_le(
    a vtl_time_period, b vtl_time_period
) AS (
    CASE WHEN a IS NULL OR b IS NULL THEN NULL
         WHEN NOT vtl_period_check_indicator(a, b) THEN NULL
         ELSE a <= b END
);

CREATE OR REPLACE MACRO vtl_period_gt(
    a vtl_time_period, b vtl_time_period
) AS (
    CASE WHEN a IS NULL OR b IS NULL THEN NULL
         WHEN NOT vtl_period_check_indicator(a, b) THEN NULL
         ELSE a > b END
);

CREATE OR REPLACE MACRO vtl_period_ge(
    a vtl_time_period, b vtl_time_period
) AS (
    CASE WHEN a IS NULL OR b IS NULL THEN NULL
         WHEN NOT vtl_period_check_indicator(a, b) THEN NULL
         ELSE a >= b END
);


-- ============================================================================
-- OUTPUT REPRESENTATION MACROS: VARCHAR -> VARCHAR
-- ============================================================================
-- Convert canonical internal VARCHAR to external representation format.

-- Helper: day-of-year + year -> YYYY-MM-DD
CREATE OR REPLACE MACRO vtl_doy_to_date(year_str VARCHAR, doy INTEGER) AS (
    CAST(CAST(CAST(year_str || '-01-01' AS DATE)
         + INTERVAL (doy - 1) DAY AS DATE) AS VARCHAR)
);

-- VTL: YYYY, YYYYSn, YYYYQn, YYYYMm, YYYYWw, YYYYDd (no hyphens)
CREATE OR REPLACE MACRO vtl_period_to_vtl(input VARCHAR) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN LENGTH(input) <= 5 THEN SUBSTR(input, 1, 4)
        ELSE SUBSTR(input, 1, 4) || SUBSTR(input, 6, 1)
             || CAST(TRY_CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR)
    END
);

-- SDMX Reporting: YYYY-A1, YYYY-Ss, YYYY-Qq, YYYY-Mmm, YYYY-Www, YYYY-Dddd
CREATE OR REPLACE MACRO vtl_period_to_sdmx_reporting(input VARCHAR) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN LENGTH(input) <= 5 THEN SUBSTR(input, 1, 4) || '-A1'
        ELSE input
    END
);

-- SDMX Gregorian: YYYY, YYYY-MM, YYYY-MM-DD (only A, M, D)
CREATE OR REPLACE MACRO vtl_period_to_sdmx_gregorian(input VARCHAR) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN LENGTH(input) <= 5 THEN SUBSTR(input, 1, 4)
        WHEN SUBSTR(input, 6, 1) = 'M' THEN
            SUBSTR(input, 1, 4) || '-' || SUBSTR(input, 7)
        WHEN SUBSTR(input, 6, 1) = 'D' THEN
            vtl_doy_to_date(SUBSTR(input, 1, 4), TRY_CAST(SUBSTR(input, 7) AS INTEGER))
        ELSE
            error('VTL Error 2-1-19-21: SDMX Gregorian only supports A, M, D '
                  || 'indicators, got ' || SUBSTR(input, 6, 1))
    END
);

-- Natural: YYYY, YYYY-Sx, YYYY-Qx, YYYY-MM, YYYY-Wxx, YYYY-MM-DD
CREATE OR REPLACE MACRO vtl_period_to_natural(input VARCHAR) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN LENGTH(input) <= 5 THEN SUBSTR(input, 1, 4)
        WHEN SUBSTR(input, 6, 1) = 'M' THEN
            SUBSTR(input, 1, 4) || '-' || SUBSTR(input, 7)
        WHEN SUBSTR(input, 6, 1) = 'D' THEN
            vtl_doy_to_date(SUBSTR(input, 1, 4), TRY_CAST(SUBSTR(input, 7) AS INTEGER))
        WHEN SUBSTR(input, 6, 1) = 'W' THEN input
        ELSE
            SUBSTR(input, 1, 4) || '-' || SUBSTR(input, 6, 1)
            || CAST(TRY_CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR)
    END
);


-- =========================================================================
-- VTL String Functions
-- =========================================================================

-- VTL instr(string, pattern, start, occurrence)
CREATE OR REPLACE MACRO vtl_instr(
    s VARCHAR, pat VARCHAR, start_pos_raw BIGINT, occur_raw BIGINT
) AS (
    CASE
        WHEN s IS NULL THEN NULL
        WHEN pat IS NULL THEN NULL
        WHEN COALESCE(occur_raw, 1) = 1 THEN
            CASE
                WHEN INSTR(s[COALESCE(start_pos_raw, 1):], pat) = 0 THEN 0
                ELSE INSTR(s[COALESCE(start_pos_raw, 1):], pat)
                     + COALESCE(start_pos_raw, 1) - 1
            END
        ELSE (
            WITH RECURSIVE find_occ(pos, n) AS (
                SELECT
                    CASE WHEN INSTR(s[COALESCE(start_pos_raw, 1):], pat) = 0
                         THEN 0
                         ELSE INSTR(s[COALESCE(start_pos_raw, 1):], pat)
                              + COALESCE(start_pos_raw, 1) - 1
                    END,
                    1
                UNION ALL
                SELECT
                    CASE WHEN pos = 0 THEN 0
                         WHEN INSTR(s[pos + 1:], pat) = 0 THEN 0
                         ELSE INSTR(s[pos + 1:], pat) + pos
                    END,
                    n + 1
                FROM find_occ
                WHERE n < COALESCE(occur_raw, 1) AND pos > 0
            )
            SELECT COALESCE(
                MAX(CASE WHEN n = COALESCE(occur_raw, 1) THEN pos END), 0
            ) FROM find_occ
        )
    END
);


-- Division that mirrors VTL error 2-1-15-6: raise when denominator is 0.
CREATE OR REPLACE MACRO vtl_div(a, b) AS (
    CASE WHEN b = 0 THEN error('VTL 2-1-15-6: Scalar division by Zero') ELSE a / b END
);
