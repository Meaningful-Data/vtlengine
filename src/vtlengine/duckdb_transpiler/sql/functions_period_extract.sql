-- TimePeriod Extraction Functions

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
