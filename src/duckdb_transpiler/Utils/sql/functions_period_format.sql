-- TimePeriod Format Function
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
