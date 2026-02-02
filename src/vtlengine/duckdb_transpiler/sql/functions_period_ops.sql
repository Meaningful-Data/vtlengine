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
