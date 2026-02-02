-- TimePeriod Operation Functions

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
-- Uses inline calculations to avoid subquery issues with MACRO parameters
CREATE OR REPLACE MACRO vtl_period_shift(p, n) AS (
    CASE
        WHEN p IS NULL THEN NULL
        WHEN p.period_indicator = 'A' THEN
            -- Annual: just add years
            vtl_period_parse(CAST(YEAR(p.start_date) + n AS VARCHAR))
        WHEN p.period_indicator = 'S' THEN
            -- Semester: 2 per year
            -- year = base_year + floor((semester + shift - 1) / 2)
            -- new_semester = ((semester + shift - 1) % 2 + 2) % 2 + 1
            vtl_period_parse(
                CAST(
                    YEAR(p.start_date) +
                    CAST(FLOOR((CAST(CEIL(MONTH(p.start_date) / 6.0) AS INTEGER) + n - 1) / 2.0) AS INTEGER)
                AS VARCHAR) || '-S' ||
                CAST(
                    ((CAST(CEIL(MONTH(p.start_date) / 6.0) AS INTEGER) + n - 1) % 2 + 2) % 2 + 1
                AS VARCHAR)
            )
        WHEN p.period_indicator = 'Q' THEN
            -- Quarter: 4 per year
            -- year = base_year + floor((quarter + shift - 1) / 4)
            -- new_quarter = ((quarter + shift - 1) % 4 + 4) % 4 + 1
            vtl_period_parse(
                CAST(
                    YEAR(p.start_date) +
                    CAST(FLOOR((QUARTER(p.start_date) + n - 1) / 4.0) AS INTEGER)
                AS VARCHAR) || '-Q' ||
                CAST(
                    ((QUARTER(p.start_date) + n - 1) % 4 + 4) % 4 + 1
                AS VARCHAR)
            )
        WHEN p.period_indicator = 'M' THEN
            -- Month: 12 per year
            -- year = base_year + floor((month + shift - 1) / 12)
            -- new_month = ((month + shift - 1) % 12 + 12) % 12 + 1
            vtl_period_parse(
                CAST(
                    YEAR(p.start_date) +
                    CAST(FLOOR((MONTH(p.start_date) + n - 1) / 12.0) AS INTEGER)
                AS VARCHAR) || '-M' ||
                LPAD(CAST(
                    ((MONTH(p.start_date) + n - 1) % 12 + 12) % 12 + 1
                AS VARCHAR), 2, '0')
            )
        WHEN p.period_indicator = 'W' THEN
            -- Week: use date arithmetic
            vtl_period_parse(
                CAST(YEAR(p.start_date + INTERVAL (n * 7) DAY) AS VARCHAR) || '-W' ||
                LPAD(CAST(WEEKOFYEAR(p.start_date + INTERVAL (n * 7) DAY) AS VARCHAR), 2, '0')
            )
        WHEN p.period_indicator = 'D' THEN
            -- Day: use date arithmetic
            vtl_period_parse(
                CAST(YEAR(p.start_date + INTERVAL (n) DAY) AS VARCHAR) || '-D' ||
                LPAD(CAST(DAYOFYEAR(p.start_date + INTERVAL (n) DAY) AS VARCHAR), 3, '0')
            )
    END
);

-- Difference in days between two TimePeriods (uses end_date per VTL spec)
CREATE OR REPLACE MACRO vtl_period_diff(a, b) AS (
    CASE
        WHEN a IS NULL OR b IS NULL THEN NULL
        ELSE ABS(DATE_DIFF('day', a.end_date, b.end_date))
    END
);
