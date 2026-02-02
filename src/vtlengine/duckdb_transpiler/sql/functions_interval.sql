-- TimeInterval Functions
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

-- Get interval length in days
CREATE OR REPLACE MACRO vtl_interval_days(i) AS (
    CASE
        WHEN i IS NULL THEN NULL
        ELSE DATE_DIFF('day', i.start_date, i.end_date)
    END
);

-- Sort key for TimeInterval (for ORDER BY and aggregations)
CREATE OR REPLACE MACRO vtl_interval_sort_key(i) AS (
    CASE
        WHEN i IS NULL THEN NULL
        ELSE [EPOCH_DAYS(i.start_date), EPOCH_DAYS(i.end_date)]
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
