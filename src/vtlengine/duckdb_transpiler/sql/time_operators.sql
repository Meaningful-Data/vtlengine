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
    CASE
        WHEN days IS NULL THEN NULL
        WHEN days < 0 THEN error('vtl error 2-1-19-16: negative value for daytoyear')
        ELSE 'P' || CAST(days // 365 AS VARCHAR) || 'Y' || CAST(days % 365 AS VARCHAR) || 'D'
    END
);

CREATE OR REPLACE MACRO vtl_daytomonth(days) AS (
    CASE
        WHEN days IS NULL THEN NULL
        WHEN days < 0 THEN error('vtl error 2-1-19-16: negative value for daytomonth')
        ELSE 'P' || CAST(days // 30 AS VARCHAR) || 'M' || CAST(days % 30 AS VARCHAR) || 'D'
    END
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

-- Map period indicator to numeric rank (higher = coarser)
CREATE OR REPLACE MACRO vtl_period_rank(ind VARCHAR) AS (
    CASE ind
        WHEN 'A' THEN 6  WHEN 'S' THEN 5  WHEN 'Q' THEN 4
        WHEN 'M' THEN 3  WHEN 'W' THEN 2  WHEN 'D' THEN 1
        ELSE 0
    END
);

-- TimePeriod → TimePeriod (convert via end_date)
CREATE OR REPLACE MACRO vtl_time_agg_tp(p vtl_time_period, target VARCHAR) AS (
    CASE
        WHEN vtl_period_rank(p.period_indicator) > vtl_period_rank(target)
            THEN error('VTL Error 2-1-19-1: Cannot aggregate period indicator '
                || p.period_indicator || ' to finer target ' || target)
        WHEN p.period_indicator = target THEN vtl_period_to_string(p)
        ELSE vtl_time_agg_date(vtl_tp_end_date(p), target)
    END
);


-- ============================================================================
-- Calendar-aware helpers frequency inference for Date timeshift
-- ============================================================================

CREATE OR REPLACE MACRO vtl_date_gap_clean_month(d_prev, d_next) AS (
    DATE_DIFF('month', d_prev, d_next) > 0
    AND (
        CAST(d_prev + to_months(CAST(DATE_DIFF('month', d_prev, d_next) AS INTEGER)) AS DATE)
            = CAST(d_next AS DATE)
        OR (
            CAST(d_prev AS DATE) = LAST_DAY(CAST(d_prev AS DATE))
            AND CAST(d_next AS DATE) = LAST_DAY(CAST(d_next AS DATE))
        )
    )
);


CREATE OR REPLACE MACRO vtl_date_timeshift_period_ind(
    n,
    has_dirty_gap,
    all_days_weekly,
    all_months_annual,
    all_months_semester,
    all_months_quarter
) AS (
    CASE
        WHEN n = 0 THEN 'A'
        WHEN has_dirty_gap THEN
            CASE WHEN all_days_weekly THEN 'W' ELSE 'D' END
        WHEN all_months_annual THEN 'A'
        WHEN all_months_semester THEN 'S'
        WHEN all_months_quarter THEN 'Q'
        ELSE 'M'
    END
);

CREATE OR REPLACE MACRO vtl_period_ind_to_interval(period_ind) AS (
    CASE period_ind
        WHEN 'D' THEN INTERVAL 1 DAY
        WHEN 'W' THEN INTERVAL 7 DAY
        WHEN 'M' THEN INTERVAL 1 MONTH
        WHEN 'Q' THEN INTERVAL 3 MONTH
        WHEN 'S' THEN INTERVAL 6 MONTH
        WHEN 'A' THEN INTERVAL 1 YEAR
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


-- ============================================================================
-- TIMEINTERVAL FREQUENCY AND SHIFT
-- ============================================================================
-- The frequency of a TimeInterval series is the DURATION of a single interval,
-- not a calendar anchor, so vtl_interval_to_period is not an equivalent here.
-- Reference: Time._classify_interval_period in Operators/Time.py.

-- The endpoints. vtl_interval_parse cannot be reused: it assumes the date part
-- is exactly 10 characters, which breaks on the 2020-01-01T00:00:00/... form.
CREATE OR REPLACE MACRO vtl_interval_start_date(s) AS (
    CAST(SUBSTR(SPLIT_PART(s, '/', 1), 1, 10) AS DATE)
);

CREATE OR REPLACE MACRO vtl_interval_end_date(s) AS (
    CAST(SUBSTR(SPLIT_PART(s, '/', 2), 1, 10) AS DATE)
);

-- Whole calendar months from d1 to c: the largest m with d1 + m months <= c.
CREATE OR REPLACE MACRO vtl_interval_months(d1, c) AS (
    date_diff('month', d1, c)
    - CASE WHEN d1 + INTERVAL (date_diff('month', d1, c)) MONTH > c THEN 1 ELSE 0 END
);

-- The days left over once those whole months are taken out. Together with
-- vtl_interval_months this reproduces relativedelta's normalisation.
CREATE OR REPLACE MACRO vtl_interval_days(d1, c) AS (
    date_diff('day', d1 + INTERVAL (vtl_interval_months(d1, c)) MONTH, c)
);

-- The six canonical VTL frequencies as (months, days): Y S Q M W D.
CREATE OR REPLACE MACRO vtl_interval_is_canonical(m, d) AS (
    (d = 0 AND m IN (12, 6, 3, 1)) OR (m = 0 AND d IN (7, 1))
);

-- Non-zero component count, over relativedelta's (years, months, days) split.
CREATE OR REPLACE MACRO vtl_interval_nonzero(m, d) AS (
    CASE WHEN m // 12 <> 0 THEN 1 ELSE 0 END
    + CASE WHEN m % 12 <> 0 THEN 1 ELSE 0 END
    + CASE WHEN d <> 0 THEN 1 ELSE 0 END
);

-- The frequency of one interval as a (months, days) STRUCT. Both the interval
-- end and end + 1 day are candidates, the first canonical one wins, and
-- otherwise the one with fewer non-zero components does, ties going to the end
-- itself. A STRUCT rather than an INTERVAL because DuckDB normalises a month to
-- 30 days when comparing intervals, which would equate 30 days with one month.
CREATE OR REPLACE MACRO vtl_interval_freq(s) AS ((
    SELECT CASE
        WHEN (m1 > 0 OR d1 > 0) AND vtl_interval_is_canonical(m1, d1)
            THEN {'months': m1, 'days': d1}
        WHEN vtl_interval_is_canonical(m2, d2) THEN {'months': m2, 'days': d2}
        -- A zero-length candidate is dropped, as relativedelta's filter does.
        WHEN m1 = 0 AND d1 = 0 THEN {'months': m2, 'days': d2}
        WHEN vtl_interval_nonzero(m1, d1) <= vtl_interval_nonzero(m2, d2)
            THEN {'months': m1, 'days': d1}
        ELSE {'months': m2, 'days': d2}
    END
    FROM (SELECT vtl_interval_months(a, b) AS m1,
                 vtl_interval_days(a, b) AS d1,
                 vtl_interval_months(a, b + INTERVAL 1 DAY) AS m2,
                 vtl_interval_days(a, b + INTERVAL 1 DAY) AS d2
          FROM (SELECT vtl_interval_start_date(s) AS a,
                       vtl_interval_end_date(s) AS b) AS _iv) AS _cand
));

-- The same frequency as a native step, for generate_series and date arithmetic.
CREATE OR REPLACE MACRO vtl_interval_freq_to_step(f) AS (
    INTERVAL (f.months) MONTH + INTERVAL (f.days) DAY
);

CREATE OR REPLACE MACRO vtl_interval_step(s) AS (
    vtl_interval_freq_to_step(vtl_interval_freq(s))
);

-- Shift both endpoints by n periods, mirroring Time_Shift.shift_interval: plain
-- calendar addition (pd.DateOffset clamps but never snaps to a month end), with
-- the output format taken from the start endpoint. STRFTIME needs a literal
-- format, hence the duplicated arms.
CREATE OR REPLACE MACRO vtl_interval_shift(s, n, step) AS (
    CASE
        WHEN s IS NULL THEN NULL
        ELSE (SELECT CASE
            WHEN LENGTH(SPLIT_PART(s, '/', 1)) > 10
                THEN STRFTIME(a + step * n, '%Y-%m-%dT%H:%M:%S') || '/'
                     || STRFTIME(b + step * n, '%Y-%m-%dT%H:%M:%S')
            ELSE STRFTIME(a + step * n, '%Y-%m-%d') || '/'
                 || STRFTIME(b + step * n, '%Y-%m-%d')
        END
        FROM (SELECT CAST(SPLIT_PART(s, '/', 1) AS TIMESTAMP) AS a,
                     CAST(SPLIT_PART(s, '/', 2) AS TIMESTAMP) AS b) AS _iv)
    END
);
