-- ============================================================================
-- VTL Time Types for DuckDB
-- ============================================================================
-- Types and macros for TimePeriod and TimeInterval handling.
-- Loaded once when initializing a DuckDB connection for VTL.
--
-- Architecture:
-- 1. vtl_period_normalize: Any input VARCHAR -> canonical internal VARCHAR
-- 2. vtl_period_parse / vtl_period_to_string: Internal VARCHAR <-> STRUCT
-- 3. vtl_period_lt/le/gt/ge: STRUCT-based ordering with same-indicator check
-- 4. Equality (=, <>): native VARCHAR comparison (no macros needed)
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
-- NORMALIZE: Any input format (#505) -> canonical internal VARCHAR
-- ============================================================================
-- Runs once at data load time. All subsequent operations use the normalized form.
-- Reference: from_input_customer_support_to_internal (TimeHandling.py:79-110)

CREATE OR REPLACE MACRO vtl_period_normalize(input) AS (
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
-- PARSE: Internal VARCHAR -> vtl_time_period STRUCT
-- ============================================================================
-- Only handles the canonical format from TimePeriodHandler.__str__

CREATE OR REPLACE MACRO vtl_period_parse(input) AS (
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
-- FORMAT: vtl_time_period STRUCT -> internal VARCHAR
-- ============================================================================
-- Reference: TimePeriodHandler.__str__ (TimeHandling.py:173-182)

CREATE OR REPLACE MACRO vtl_period_to_string(p) AS (
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

CREATE OR REPLACE MACRO vtl_interval_parse(input) AS (
    CASE
        WHEN input IS NULL THEN NULL
        ELSE {
            'date1': CAST(SUBSTR(input, 1, 10) AS DATE),
            'date2': CAST(SUBSTR(input, 12) AS DATE)
        }::vtl_time_interval
    END
);

CREATE OR REPLACE MACRO vtl_interval_to_string(i) AS (
    CASE
        WHEN i IS NULL THEN NULL
        ELSE CAST(i.date1 AS VARCHAR) || '/' || CAST(i.date2 AS VARCHAR)
    END
);


-- ============================================================================
-- COMPARISON MACROS (ordering only -- equality uses VARCHAR directly)
-- ============================================================================

CREATE OR REPLACE MACRO vtl_period_check_indicator(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN TRUE
        WHEN a.period_indicator != b.period_indicator THEN
            error('VTL Error 2-1-19-19: Cannot compare TimePeriods with different indicators: '
                  || a.period_indicator || ' vs ' || b.period_indicator)
        ELSE TRUE
    END
);

CREATE OR REPLACE MACRO vtl_period_lt(a, b) AS (
    CASE WHEN a IS NULL OR b IS NULL THEN NULL
         WHEN NOT vtl_period_check_indicator(a, b) THEN NULL
         ELSE a < b END
);

CREATE OR REPLACE MACRO vtl_period_le(a, b) AS (
    CASE WHEN a IS NULL OR b IS NULL THEN NULL
         WHEN NOT vtl_period_check_indicator(a, b) THEN NULL
         ELSE a <= b END
);

CREATE OR REPLACE MACRO vtl_period_gt(a, b) AS (
    CASE WHEN a IS NULL OR b IS NULL THEN NULL
         WHEN NOT vtl_period_check_indicator(a, b) THEN NULL
         ELSE a > b END
);

CREATE OR REPLACE MACRO vtl_period_ge(a, b) AS (
    CASE WHEN a IS NULL OR b IS NULL THEN NULL
         WHEN NOT vtl_period_check_indicator(a, b) THEN NULL
         ELSE a >= b END
);

-- ============================================================================
-- OUTPUT REPRESENTATION MACROS (VARCHAR -> VARCHAR only)
-- ============================================================================
-- Convert canonical internal VARCHAR to external representation format.
-- Input is always the canonical format (e.g. '2020-M06', '2020A').
-- All macros operate on VARCHAR only, no STRUCT or DATE types.
-- NULL inputs are handled by COALESCE/IF — no explicit CASE WHEN NULL checks.

-- Helper: convert day-of-year (1-366) + year to YYYY-MM-DD using VARCHAR only.
CREATE OR REPLACE MACRO vtl_doy_to_date(year_str, doy) AS (
    SELECT year_str || '-'
           || LPAD(CAST(m.month_num AS VARCHAR), 2, '0') || '-'
           || LPAD(CAST(doy - m.month_start AS VARCHAR), 2, '0')
    FROM (
        SELECT months.m AS month_num,
               IF(CAST(year_str AS INTEGER) % 4 = 0
                  AND (CAST(year_str AS INTEGER) % 100 != 0
                       OR CAST(year_str AS INTEGER) % 400 = 0),
                  months.cum_leap, months.cum) AS month_start
        FROM (
            SELECT unnest([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]) AS cum,
                   unnest([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]) AS cum_leap,
                   unnest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) AS m
        ) months
        WHERE IF(CAST(year_str AS INTEGER) % 4 = 0
                 AND (CAST(year_str AS INTEGER) % 100 != 0
                      OR CAST(year_str AS INTEGER) % 400 = 0),
                 months.cum_leap, months.cum) < doy
        ORDER BY months.m DESC
        LIMIT 1
    ) m
);

-- VTL: YYYY, YYYYSn, YYYYQn, YYYYMm, YYYYWw, YYYYDd (no hyphens)
CREATE OR REPLACE MACRO vtl_period_to_vtl(input) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN LENGTH(input) <= 5 THEN SUBSTR(input, 1, 4)
        ELSE SUBSTR(input, 1, 4) || SUBSTR(input, 6, 1)
             || CAST(TRY_CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR)
    END
);

-- SDMX Reporting: YYYY-A1, YYYY-Ss, YYYY-Qq, YYYY-Mmm, YYYY-Www, YYYY-Dddd
CREATE OR REPLACE MACRO vtl_period_to_sdmx_reporting(input) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN LENGTH(input) <= 5 THEN SUBSTR(input, 1, 4) || '-A1'
        ELSE input
    END
);

-- SDMX Gregorian: YYYY, YYYY-MM, YYYY-MM-DD (only A, M, D)
CREATE OR REPLACE MACRO vtl_period_to_sdmx_gregorian(input) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN LENGTH(input) <= 5 THEN SUBSTR(input, 1, 4)
        WHEN SUBSTR(input, 6, 1) = 'M' THEN
            SUBSTR(input, 1, 4) || '-' || SUBSTR(input, 7)
        WHEN SUBSTR(input, 6, 1) = 'D' THEN
            vtl_doy_to_date(SUBSTR(input, 1, 4), TRY_CAST(SUBSTR(input, 7) AS INTEGER))
        ELSE
            error('VTL Error 2-1-19-21: SDMX Gregorian only supports A, M, D indicators, got '
                  || SUBSTR(input, 6, 1))
    END
);

-- Natural: YYYY, YYYY-Sx, YYYY-Qx, YYYY-MM, YYYY-Wxx, YYYY-MM-DD
CREATE OR REPLACE MACRO vtl_period_to_natural(input) AS (
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
CREATE OR REPLACE MACRO vtl_instr(s, pat, start_pos_raw, occur_raw) AS (
    CASE
        WHEN s IS NULL THEN NULL
        WHEN pat IS NULL THEN NULL
        WHEN COALESCE(occur_raw, 1) = 1 THEN
            CASE
                WHEN INSTR(s[COALESCE(start_pos_raw, 1):], pat) = 0 THEN 0
                ELSE INSTR(s[COALESCE(start_pos_raw, 1):], pat) + COALESCE(start_pos_raw, 1) - 1
            END
        ELSE (
            WITH RECURSIVE find_occ(pos, n) AS (
                SELECT
                    CASE WHEN INSTR(s[COALESCE(start_pos_raw, 1):], pat) = 0 THEN 0
                         ELSE INSTR(s[COALESCE(start_pos_raw, 1):], pat) + COALESCE(start_pos_raw, 1) - 1
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
            SELECT COALESCE(MAX(CASE WHEN n = COALESCE(occur_raw, 1) THEN pos END), 0) FROM find_occ
        )
    END
);
