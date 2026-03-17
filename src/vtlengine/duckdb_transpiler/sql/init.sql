-- ============================================================================
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
        WHEN LENGTH(input) = 10 THEN
            SUBSTR(input, 1, 4) || '-D'
            || LPAD(CAST(DAYOFYEAR(CAST(input AS DATE)) AS VARCHAR), 3, '0')
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
    s VARCHAR, pat VARCHAR, start_pos_raw INTEGER, occur_raw INTEGER
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
