-- VTL Time Types for DuckDB
-- TimePeriod: Regular periods like 2022Q3, 2022-M01, 2022-S02
-- TimeInterval: Date intervals like 2021-01-01/2022-01-01

-- Drop existing types if they exist (for development)
DROP TYPE IF EXISTS vtl_time_period;
DROP TYPE IF EXISTS vtl_time_interval;

-- TimePeriod STRUCT: stores date range and period indicator
CREATE TYPE vtl_time_period AS STRUCT(
    start_date DATE,
    end_date DATE,
    period_indicator VARCHAR
);

-- TimeInterval STRUCT: stores date range
CREATE TYPE vtl_time_interval AS STRUCT(
    start_date DATE,
    end_date DATE
);
