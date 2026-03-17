-- VTL Time Types for DuckDB
-- TimePeriod: Regular periods like 2022Q3, 2022-M01, 2022-S02
-- TimeInterval: Date intervals like 2021-01-01/2022-01-01

-- Drop existing types if they exist (for development)
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
