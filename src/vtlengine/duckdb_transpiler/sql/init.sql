-- ============================================================================
-- VTL Time Types for DuckDB - Combined Initialization Script
-- ============================================================================
-- This file contains all SQL definitions for VTL time types in DuckDB.
-- It should be loaded once when initializing a DuckDB connection for VTL.
--
-- Contents:
-- 1. Type definitions (vtl_time_period, vtl_time_interval)
-- 2. TimePeriod parse functions
-- 3. TimePeriod format functions
-- 4. TimePeriod comparison functions
-- 5. TimePeriod extraction functions
-- 6. TimePeriod operation functions (shift, diff, time_agg)
-- 7. TimeInterval functions
-- ============================================================================


-- ============================================================================
-- TYPE DEFINITIONS
-- ============================================================================
-- TimePeriod: Regular periods like 2022Q3, 2022-M01, 2022-S02
-- TimeInterval: Date intervals like 2021-01-01/2022-01-01

-- Drop existing types if they exist (for development)
DROP TYPE IF EXISTS vtl_time_period;
DROP TYPE IF EXISTS vtl_time_interval;

-- TimePeriod STRUCT: stores date range and period indicator
CREATE TYPE vtl_time_period AS STRUCT(
    start_date DATE,
    end_date DATE,
    period_indicator VARCHAR
);

-- TimeInterval STRUCT: stores date range
CREATE TYPE vtl_time_interval AS STRUCT(
    start_date DATE,
    end_date DATE
);


-- ============================================================================
-- TIMEPERIOD PARSE FUNCTIONS
-- ============================================================================
-- Parses VTL TimePeriod strings to vtl_time_period STRUCT
-- Handles formats: 2022, 2022A, 2022-Q3, 2022Q3, 2022-M06, 2022M06, etc.

CREATE OR REPLACE MACRO vtl_period_parse(input) AS (
    CASE
        WHEN input IS NULL THEN NULL
        ELSE (
            WITH parsed AS (
                SELECT
                    -- Extract year (always first 4 chars)
                    CAST(LEFT(TRIM(input), 4) AS INTEGER) AS year,
                    -- Extract indicator and number from rest
                    CASE
                        -- Just year: '2022' -> Annual
                        WHEN LENGTH(TRIM(input)) = 4 THEN 'A'
                        -- With dash: '2022-Q3' or '2022-M06'
                        WHEN SUBSTRING(TRIM(input), 5, 1) = '-' THEN UPPER(SUBSTRING(TRIM(input), 6, 1))
                        -- Without dash: '2022Q3' or '2022M06' or '2022A'
                        ELSE UPPER(SUBSTRING(TRIM(input), 5, 1))
                    END AS indicator,
                    CASE
                        -- Annual: no number needed
                        WHEN LENGTH(TRIM(input)) = 4 THEN 1
                        WHEN LENGTH(TRIM(input)) = 5 AND UPPER(SUBSTRING(TRIM(input), 5, 1)) = 'A' THEN 1
                        -- With dash: '2022-Q3' -> 3, '2022-M06' -> 6
                        WHEN SUBSTRING(TRIM(input), 5, 1) = '-' THEN
                            CAST(SUBSTRING(TRIM(input), 7) AS INTEGER)
                        -- Without dash: '2022Q3' -> 3, '2022M06' -> 6
                        ELSE CAST(SUBSTRING(TRIM(input), 6) AS INTEGER)
                    END AS number
            )
            SELECT {
                'start_date': CASE parsed.indicator
                    WHEN 'A' THEN MAKE_DATE(parsed.year, 1, 1)
                    WHEN 'S' THEN MAKE_DATE(parsed.year, (parsed.number - 1) * 6 + 1, 1)
                    WHEN 'Q' THEN MAKE_DATE(parsed.year, (parsed.number - 1) * 3 + 1, 1)
                    WHEN 'M' THEN MAKE_DATE(parsed.year, parsed.number, 1)
                    WHEN 'W' THEN CAST(
                        STRPTIME(parsed.year || '-W' || LPAD(CAST(parsed.number AS VARCHAR), 2, '0') || '-1', '%G-W%V-%u')
                        AS DATE
                    )
                    WHEN 'D' THEN CAST(
                        STRPTIME(parsed.year || '-' || LPAD(CAST(parsed.number AS VARCHAR), 3, '0'), '%Y-%j')
                        AS DATE
                    )
                END,
                'end_date': CASE parsed.indicator
                    WHEN 'A' THEN MAKE_DATE(parsed.year, 12, 31)
                    WHEN 'S' THEN LAST_DAY(MAKE_DATE(parsed.year, parsed.number * 6, 1))
                    WHEN 'Q' THEN LAST_DAY(MAKE_DATE(parsed.year, parsed.number * 3, 1))
                    WHEN 'M' THEN LAST_DAY(MAKE_DATE(parsed.year, parsed.number, 1))
                    WHEN 'W' THEN CAST(
                        STRPTIME(parsed.year || '-W' || LPAD(CAST(parsed.number AS VARCHAR), 2, '0') || '-7', '%G-W%V-%u')
                        AS DATE
                    )
                    WHEN 'D' THEN CAST(
                        STRPTIME(parsed.year || '-' || LPAD(CAST(parsed.number AS VARCHAR), 3, '0'), '%Y-%j')
                        AS DATE
                    )
                END,
                'period_indicator': parsed.indicator
            }::vtl_time_period
            FROM parsed
        )
    END
);


-- ============================================================================
-- TIMEPERIOD FORMAT FUNCTIONS
-- ============================================================================
-- Formats vtl_time_period STRUCT back to VTL string format
-- Output: 2022, 2022-S1, 2022-Q3, 2022-M06, 2022-W15, 2022-D100

CREATE OR REPLACE MACRO vtl_period_to_string(p) AS (
    CASE p.period_indicator
        WHEN 'A' THEN CAST(YEAR(CAST(p.start_date AS DATE)) AS VARCHAR)
        WHEN 'S' THEN
            CAST(YEAR(CAST(p.start_date AS DATE)) AS VARCHAR) || '-S' ||
            CAST(CAST(CEIL(MONTH(CAST(p.start_date AS DATE)) / 6.0) AS INTEGER) AS VARCHAR)
        WHEN 'Q' THEN
            CAST(YEAR(CAST(p.start_date AS DATE)) AS VARCHAR) || '-Q' ||
            CAST(QUARTER(CAST(p.start_date AS DATE)) AS VARCHAR)
        WHEN 'M' THEN
            CAST(YEAR(CAST(p.start_date AS DATE)) AS VARCHAR) || '-M' ||
            LPAD(CAST(MONTH(CAST(p.start_date AS DATE)) AS VARCHAR), 2, '0')
        WHEN 'W' THEN
            CAST(YEAR(CAST(p.start_date AS DATE)) AS VARCHAR) || '-W' ||
            LPAD(CAST(WEEKOFYEAR(CAST(p.start_date AS DATE)) AS VARCHAR), 2, '0')
        WHEN 'D' THEN
            CAST(YEAR(CAST(p.start_date AS DATE)) AS VARCHAR) || '-D' ||
            LPAD(CAST(DAYOFYEAR(CAST(p.start_date AS DATE)) AS VARCHAR), 3, '0')
        ELSE NULL
    END
);


-- ============================================================================
-- TIMEPERIOD COMPARISON FUNCTIONS
-- ============================================================================
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


-- ============================================================================
-- TIMEPERIOD EXTRACTION FUNCTIONS
-- ============================================================================

-- Extract year
CREATE OR REPLACE MACRO vtl_period_year(p) AS (
    CASE WHEN p IS NULL THEN CAST(NULL AS INTEGER) ELSE YEAR(CAST(p.start_date AS DATE)) END
);

-- Extract period indicator
CREATE OR REPLACE MACRO vtl_period_indicator(p) AS (
    CASE WHEN p IS NULL THEN CAST(NULL AS VARCHAR) ELSE p.period_indicator END
);

-- Extract period number within year
CREATE OR REPLACE MACRO vtl_period_number(p) AS (
    CASE
        WHEN p IS NULL THEN CAST(NULL AS INTEGER)
        WHEN p.period_indicator = 'A' THEN 1
        WHEN p.period_indicator = 'S' THEN CAST(CEIL(MONTH(CAST(p.start_date AS DATE)) / 6.0) AS INTEGER)
        WHEN p.period_indicator = 'Q' THEN QUARTER(CAST(p.start_date AS DATE))
        WHEN p.period_indicator = 'M' THEN MONTH(CAST(p.start_date AS DATE))
        WHEN p.period_indicator = 'W' THEN WEEKOFYEAR(CAST(p.start_date AS DATE))
        WHEN p.period_indicator = 'D' THEN DAYOFYEAR(CAST(p.start_date AS DATE))
    END
);


-- ============================================================================
-- TIMEPERIOD OPERATION FUNCTIONS
-- ============================================================================

-- Period limits per indicator
CREATE OR REPLACE MACRO vtl_period_limit(indicator) AS (
    CASE indicator
        WHEN 'A' THEN 1
        WHEN 'S' THEN 2
        WHEN 'Q' THEN 4
        WHEN 'M' THEN 12
        WHEN 'W' THEN 52
        WHEN 'D' THEN 365
    END
);

-- Shift TimePeriod by N periods
-- Optimized: directly constructs STRUCT using date arithmetic instead of parsing strings
CREATE OR REPLACE MACRO vtl_period_shift(p, n) AS (
    CASE
        WHEN p IS NULL THEN NULL
        WHEN p.period_indicator = 'A' THEN
            -- Annual: add years directly
            {
                'start_date': MAKE_DATE(YEAR(p.start_date) + n, 1, 1),
                'end_date': MAKE_DATE(YEAR(p.start_date) + n, 12, 31),
                'period_indicator': 'A'
            }::vtl_time_period
        WHEN p.period_indicator = 'S' THEN
            -- Semester: use month arithmetic (6 months per semester)
            {
                'start_date': CAST(p.start_date + INTERVAL (n * 6) MONTH AS DATE),
                'end_date': LAST_DAY(CAST(p.start_date + INTERVAL (n * 6 + 5) MONTH AS DATE)),
                'period_indicator': 'S'
            }::vtl_time_period
        WHEN p.period_indicator = 'Q' THEN
            -- Quarter: use month arithmetic (3 months per quarter)
            {
                'start_date': CAST(p.start_date + INTERVAL (n * 3) MONTH AS DATE),
                'end_date': LAST_DAY(CAST(p.start_date + INTERVAL (n * 3 + 2) MONTH AS DATE)),
                'period_indicator': 'Q'
            }::vtl_time_period
        WHEN p.period_indicator = 'M' THEN
            -- Month: use month arithmetic directly
            {
                'start_date': CAST(p.start_date + INTERVAL (n) MONTH AS DATE),
                'end_date': LAST_DAY(CAST(p.start_date + INTERVAL (n) MONTH AS DATE)),
                'period_indicator': 'M'
            }::vtl_time_period
        WHEN p.period_indicator = 'W' THEN
            -- Week: use day arithmetic (7 days per week)
            {
                'start_date': CAST(p.start_date + INTERVAL (n * 7) DAY AS DATE),
                'end_date': CAST(p.end_date + INTERVAL (n * 7) DAY AS DATE),
                'period_indicator': 'W'
            }::vtl_time_period
        WHEN p.period_indicator = 'D' THEN
            -- Day: use day arithmetic directly
            {
                'start_date': CAST(p.start_date + INTERVAL (n) DAY AS DATE),
                'end_date': CAST(p.start_date + INTERVAL (n) DAY AS DATE),
                'period_indicator': 'D'
            }::vtl_time_period
    END
);

-- Difference in days between two TimePeriods (uses end_date per VTL spec)
CREATE OR REPLACE MACRO vtl_period_diff(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        ELSE ABS(DATE_DIFF('day', a.end_date, b.end_date))
    END
);

-- Period indicator order (higher = coarser)
CREATE OR REPLACE MACRO vtl_period_order(indicator) AS (
    CASE indicator
        WHEN 'D' THEN 1
        WHEN 'W' THEN 2
        WHEN 'M' THEN 3
        WHEN 'Q' THEN 4
        WHEN 'S' THEN 5
        WHEN 'A' THEN 6
    END
);

-- Time aggregation to coarser granularity
-- Optimized: directly constructs STRUCT instead of parsing strings
CREATE OR REPLACE MACRO vtl_time_agg(p, target_indicator) AS (
    CASE
        WHEN p IS NULL THEN NULL
        WHEN vtl_period_order(p.period_indicator) >= vtl_period_order(target_indicator) THEN
            error('VTL Error: Cannot aggregate TimePeriod from ' || p.period_indicator ||
                  ' to ' || target_indicator || '. Target must be coarser granularity.')
        WHEN target_indicator = 'A' THEN
            {
                'start_date': MAKE_DATE(YEAR(p.start_date), 1, 1),
                'end_date': MAKE_DATE(YEAR(p.start_date), 12, 31),
                'period_indicator': 'A'
            }::vtl_time_period
        WHEN target_indicator = 'S' THEN
            {
                'start_date': MAKE_DATE(YEAR(p.start_date), CASE WHEN MONTH(p.start_date) <= 6 THEN 1 ELSE 7 END, 1),
                'end_date': CASE WHEN MONTH(p.start_date) <= 6
                    THEN MAKE_DATE(YEAR(p.start_date), 6, 30)
                    ELSE MAKE_DATE(YEAR(p.start_date), 12, 31) END,
                'period_indicator': 'S'
            }::vtl_time_period
        WHEN target_indicator = 'Q' THEN
            {
                'start_date': MAKE_DATE(YEAR(p.start_date), (QUARTER(p.start_date) - 1) * 3 + 1, 1),
                'end_date': LAST_DAY(MAKE_DATE(YEAR(p.start_date), QUARTER(p.start_date) * 3, 1)),
                'period_indicator': 'Q'
            }::vtl_time_period
        WHEN target_indicator = 'M' THEN
            {
                'start_date': MAKE_DATE(YEAR(p.start_date), MONTH(p.start_date), 1),
                'end_date': LAST_DAY(MAKE_DATE(YEAR(p.start_date), MONTH(p.start_date), 1)),
                'period_indicator': 'M'
            }::vtl_time_period
        WHEN target_indicator = 'W' THEN
            {
                'start_date': DATE_TRUNC('week', p.start_date)::DATE,
                'end_date': (DATE_TRUNC('week', p.start_date) + INTERVAL 6 DAY)::DATE,
                'period_indicator': 'W'
            }::vtl_time_period
    END
);


-- ============================================================================
-- TIMEINTERVAL FUNCTIONS
-- ============================================================================
-- Parse, format, compare, and operate on date intervals

-- Parse TimeInterval string (format: 'YYYY-MM-DD/YYYY-MM-DD')
CREATE OR REPLACE MACRO vtl_interval_parse(input) AS (
    CASE
        WHEN input IS NULL THEN NULL
        ELSE {
            'start_date': CAST(SPLIT_PART(input, '/', 1) AS DATE),
            'end_date': CAST(SPLIT_PART(input, '/', 2) AS DATE)
        }::vtl_time_interval
    END
);

-- Format TimeInterval to string
CREATE OR REPLACE MACRO vtl_interval_to_string(i) AS (
    CASE
        WHEN i IS NULL THEN NULL
        ELSE CAST(i.start_date AS VARCHAR) || '/' || CAST(i.end_date AS VARCHAR)
    END
);

-- Construct TimeInterval from dates
CREATE OR REPLACE MACRO vtl_interval(start_date, end_date) AS (
    {'start_date': start_date, 'end_date': end_date}::vtl_time_interval
);

-- TimeInterval equality
CREATE OR REPLACE MACRO vtl_interval_eq(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        ELSE a.start_date = b.start_date AND a.end_date = b.end_date
    END
);

-- TimeInterval inequality
CREATE OR REPLACE MACRO vtl_interval_ne(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        ELSE a.start_date != b.start_date OR a.end_date != b.end_date
    END
);

-- TimeInterval less than (compares by start_date, then end_date)
CREATE OR REPLACE MACRO vtl_interval_lt(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        WHEN a.start_date < b.start_date THEN TRUE
        WHEN a.start_date > b.start_date THEN FALSE
        ELSE a.end_date < b.end_date
    END
);

-- TimeInterval less than or equal
CREATE OR REPLACE MACRO vtl_interval_le(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        WHEN a.start_date < b.start_date THEN TRUE
        WHEN a.start_date > b.start_date THEN FALSE
        ELSE a.end_date <= b.end_date
    END
);

-- TimeInterval greater than (compares by start_date, then end_date)
CREATE OR REPLACE MACRO vtl_interval_gt(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        WHEN a.start_date > b.start_date THEN TRUE
        WHEN a.start_date < b.start_date THEN FALSE
        ELSE a.end_date > b.end_date
    END
);

-- TimeInterval greater than or equal
CREATE OR REPLACE MACRO vtl_interval_ge(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        WHEN a.start_date > b.start_date THEN TRUE
        WHEN a.start_date < b.start_date THEN FALSE
        ELSE a.end_date >= b.end_date
    END
);

-- Get interval length in days
CREATE OR REPLACE MACRO vtl_interval_days(i) AS (
    CASE
        WHEN i IS NULL THEN NULL
        ELSE DATE_DIFF('day', i.start_date, i.end_date)
    END
);

-- Sort key for TimeInterval (for ORDER BY and aggregations)
-- Returns days since epoch for both start and end dates
CREATE OR REPLACE MACRO vtl_interval_sort_key(i) AS (
    CASE
        WHEN i IS NULL THEN NULL
        ELSE [
            (i.start_date - DATE '1970-01-01')::INTEGER,
            (i.end_date - DATE '1970-01-01')::INTEGER
        ]
    END
);

-- Shift TimeInterval by days
CREATE OR REPLACE MACRO vtl_interval_shift(i, days) AS (
    CASE
        WHEN i IS NULL THEN NULL
        ELSE {
            'start_date': i.start_date + INTERVAL (days) DAY,
            'end_date': i.end_date + INTERVAL (days) DAY
        }::vtl_time_interval
    END
);

-- =========================================================================
-- VTL String Functions
-- =========================================================================

-- VTL instr(string, pattern, start, occurrence)
-- For the simple case (start=1, occurrence=1), just use INSTR.
-- For start > 1: search in SUBSTR, add offset back.
-- For occurrence > 1: we need vtl_instr_impl which loops.
-- DuckDB macros can't do recursion, so we implement up to 10 occurrences.
CREATE OR REPLACE MACRO vtl_instr(s, pat, start_pos, occur) AS (
    CASE
        WHEN s IS NULL THEN NULL
        WHEN pat IS NULL THEN 0
        WHEN occur = 1 THEN
            CASE
                WHEN INSTR(s[start_pos:], pat) = 0 THEN 0
                ELSE INSTR(s[start_pos:], pat) + start_pos - 1
            END
        ELSE (
            -- Find nth occurrence by chaining
            WITH RECURSIVE find_occ(pos, n) AS (
                SELECT
                    CASE WHEN INSTR(s[start_pos:], pat) = 0 THEN 0
                         ELSE INSTR(s[start_pos:], pat) + start_pos - 1
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
                WHERE n < occur AND pos > 0
            )
            SELECT COALESCE(MAX(CASE WHEN n = occur THEN pos END), 0) FROM find_occ
        )
    END
);
