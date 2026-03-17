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
                    || CAST(CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR)
                WHEN UPPER(SUBSTR(input, 6, 1)) IN ('M', 'W') THEN
                    SUBSTR(input, 1, 4) || '-' || UPPER(SUBSTR(input, 6, 1))
                    || LPAD(CAST(CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR), 2, '0')
                ELSE
                    SUBSTR(input, 1, 4) || '-D'
                    || LPAD(CAST(CAST(SUBSTR(input, 7) AS INTEGER) AS VARCHAR), 3, '0')
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
-- EXTRACTION MACROS
-- ============================================================================

CREATE OR REPLACE MACRO vtl_period_year(p) AS (
    CASE WHEN p IS NULL THEN CAST(NULL AS INTEGER) ELSE p.year END
);

CREATE OR REPLACE MACRO vtl_period_indicator(p) AS (
    CASE WHEN p IS NULL THEN CAST(NULL AS VARCHAR) ELSE p.period_indicator END
);

CREATE OR REPLACE MACRO vtl_period_number(p) AS (
    CASE WHEN p IS NULL THEN CAST(NULL AS INTEGER) ELSE p.period_number END
);


-- ============================================================================
-- TIMEPERIOD SHIFT: shift a TimePeriod VARCHAR by n periods, return VARCHAR
-- ============================================================================
-- Input/output are canonical internal VARCHAR (e.g. '2020-M06').
-- Uses SUBSTR to extract components, does arithmetic, formats back.
-- Cannot use vtl_period_parse().field inside DuckDB macros, so works on VARCHAR directly.

CREATE OR REPLACE MACRO vtl_period_shift(raw_input, n) AS (
    CASE
        WHEN raw_input IS NULL THEN NULL
        ELSE (
            WITH input_str AS (
                SELECT CAST(raw_input AS VARCHAR) AS v
            ),
            parsed AS (
                SELECT
                    CASE WHEN SUBSTR(input_str.v, 5, 1) != '-' THEN 'A'
                         ELSE SUBSTR(input_str.v, 6, 1) END AS ind,
                    CAST(SUBSTR(input_str.v, 1, 4) AS INTEGER) AS y,
                    CASE WHEN SUBSTR(input_str.v, 5, 1) != '-' THEN 1
                         ELSE CAST(SUBSTR(input_str.v, 7) AS INTEGER) END AS num
                FROM input_str
            ),
            shifted AS (
                SELECT
                    parsed.ind AS ind,
                    parsed.y AS y,
                    parsed.num + n AS raw_num,
                    CASE parsed.ind
                        WHEN 'A' THEN 1 WHEN 'S' THEN 2 WHEN 'Q' THEN 4
                        WHEN 'M' THEN 12 WHEN 'W' THEN 52 WHEN 'D' THEN 365
                    END AS period_max
                FROM parsed
            )
            SELECT
                CASE shifted.ind
                    WHEN 'A' THEN
                        CAST(shifted.y + shifted.raw_num - 1 AS VARCHAR) || 'A'
                    WHEN 'S' THEN
                        CAST(shifted.y + CAST(FLOOR((shifted.raw_num - 1) / 2.0) AS INTEGER) AS VARCHAR)
                        || '-S' || CAST(((shifted.raw_num - 1) % 2 + 2) % 2 + 1 AS VARCHAR)
                    WHEN 'Q' THEN
                        CAST(shifted.y + CAST(FLOOR((shifted.raw_num - 1) / 4.0) AS INTEGER) AS VARCHAR)
                        || '-Q' || CAST(((shifted.raw_num - 1) % 4 + 4) % 4 + 1 AS VARCHAR)
                    WHEN 'M' THEN
                        CAST(shifted.y + CAST(FLOOR((shifted.raw_num - 1) / 12.0) AS INTEGER) AS VARCHAR)
                        || '-M' || LPAD(CAST(((shifted.raw_num - 1) % 12 + 12) % 12 + 1 AS VARCHAR), 2, '0')
                    WHEN 'W' THEN
                        CAST(shifted.y + CAST(FLOOR((shifted.raw_num - 1) / 52.0) AS INTEGER) AS VARCHAR)
                        || '-W' || LPAD(CAST(((shifted.raw_num - 1) % 52 + 52) % 52 + 1 AS VARCHAR), 2, '0')
                    WHEN 'D' THEN
                        CAST(shifted.y + CAST(FLOOR((shifted.raw_num - 1) / 365.0) AS INTEGER) AS VARCHAR)
                        || '-D' || LPAD(CAST(((shifted.raw_num - 1) % 365 + 365) % 365 + 1 AS VARCHAR), 3, '0')
                END
            FROM shifted
        )
    END
);


-- ============================================================================
-- TIMEPERIOD DIFF: difference between two TimePeriod VARCHARs
-- ============================================================================

CREATE OR REPLACE MACRO vtl_period_diff(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        ELSE ABS(
            (CAST(SUBSTR(a, 1, 4) AS INTEGER) * 365
             + CASE WHEN SUBSTR(a, 5, 1) = '-' THEN CAST(SUBSTR(a, 7) AS INTEGER) ELSE 1 END)
            - (CAST(SUBSTR(b, 1, 4) AS INTEGER) * 365
               + CASE WHEN SUBSTR(b, 5, 1) = '-' THEN CAST(SUBSTR(b, 7) AS INTEGER) ELSE 1 END)
        )
    END
);


-- ============================================================================
-- TIMEPERIOD OPERATIONS: time_agg, period_order
-- ============================================================================

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
-- Input/output are canonical internal VARCHAR
CREATE OR REPLACE MACRO vtl_time_agg(input, target_indicator) AS (
    CASE
        WHEN input IS NULL THEN NULL
        WHEN SUBSTR(input, 5, 1) != '-' THEN
            -- Annual input: can only aggregate to A (same)
            CASE WHEN target_indicator = 'A' THEN input
                 ELSE error('VTL Error: Cannot aggregate Annual to ' || target_indicator)
            END
        WHEN vtl_period_order(SUBSTR(input, 6, 1)) >= vtl_period_order(target_indicator) THEN
            error('VTL Error: Cannot aggregate TimePeriod from '
                  || SUBSTR(input, 6, 1) || ' to ' || target_indicator
                  || '. Target must be coarser granularity.')
        WHEN target_indicator = 'A' THEN
            SUBSTR(input, 1, 4) || 'A'
        WHEN target_indicator = 'S' THEN
            SUBSTR(input, 1, 4) || '-S'
            || CAST(
                CASE SUBSTR(input, 6, 1)
                    WHEN 'Q' THEN CASE WHEN CAST(SUBSTR(input, 7) AS INTEGER) <= 2 THEN 1 ELSE 2 END
                    WHEN 'M' THEN CASE WHEN CAST(SUBSTR(input, 7) AS INTEGER) <= 6 THEN 1 ELSE 2 END
                    WHEN 'W' THEN CASE WHEN CAST(SUBSTR(input, 7) AS INTEGER) <= 26 THEN 1 ELSE 2 END
                    WHEN 'D' THEN CASE WHEN CAST(SUBSTR(input, 7) AS INTEGER) <= 183 THEN 1 ELSE 2 END
                END AS VARCHAR)
        WHEN target_indicator = 'Q' THEN
            SUBSTR(input, 1, 4) || '-Q'
            || CAST(
                CASE SUBSTR(input, 6, 1)
                    WHEN 'M' THEN CAST(CEIL(CAST(SUBSTR(input, 7) AS INTEGER) / 3.0) AS INTEGER)
                    WHEN 'W' THEN CAST(CEIL(CAST(SUBSTR(input, 7) AS INTEGER) / 13.0) AS INTEGER)
                    WHEN 'D' THEN CAST(CEIL(CAST(SUBSTR(input, 7) AS INTEGER) / 91.25) AS INTEGER)
                END AS VARCHAR)
        WHEN target_indicator = 'M' THEN
            SUBSTR(input, 1, 4) || '-M'
            || LPAD(CAST(
                CASE SUBSTR(input, 6, 1)
                    WHEN 'W' THEN LEAST(CAST(CEIL(CAST(SUBSTR(input, 7) AS INTEGER) / 4.33) AS INTEGER), 12)
                    WHEN 'D' THEN LEAST(CAST(CEIL(CAST(SUBSTR(input, 7) AS INTEGER) / 30.44) AS INTEGER), 12)
                END AS VARCHAR), 2, '0')
        WHEN target_indicator = 'W' THEN
            SUBSTR(input, 1, 4) || '-W'
            || LPAD(CAST(
                LEAST(CAST(CEIL(CAST(SUBSTR(input, 7) AS INTEGER) / 7.0) AS INTEGER), 52)
            AS VARCHAR), 2, '0')
        ELSE NULL
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
