-- TimePeriod Parse Function
-- Parses VTL TimePeriod strings to vtl_time_period STRUCT
-- Handles formats: 2022, 2022A, 2022-Q3, 2022Q3, 2022-M06, 2022M06, etc.

CREATE OR REPLACE MACRO vtl_period_parse(input) AS (
    CASE
        WHEN input IS NULL THEN NULL
        ELSE (
            WITH parsed AS (
                SELECT
                    -- Extract year (always first 4 chars)
                    CAST(LEFT(TRIM(input), 4) AS INTEGER) AS year,
                    -- Extract indicator and number from rest
                    CASE
                        -- Just year: '2022' -> Annual
                        WHEN LENGTH(TRIM(input)) = 4 THEN 'A'
                        -- With dash: '2022-Q3' or '2022-M06'
                        WHEN SUBSTRING(TRIM(input), 5, 1) = '-' THEN UPPER(SUBSTRING(TRIM(input), 6, 1))
                        -- Without dash: '2022Q3' or '2022M06' or '2022A'
                        ELSE UPPER(SUBSTRING(TRIM(input), 5, 1))
                    END AS indicator,
                    CASE
                        -- Annual: no number needed
                        WHEN LENGTH(TRIM(input)) = 4 THEN 1
                        WHEN LENGTH(TRIM(input)) = 5 AND UPPER(SUBSTRING(TRIM(input), 5, 1)) = 'A' THEN 1
                        -- With dash: '2022-Q3' -> 3, '2022-M06' -> 6
                        WHEN SUBSTRING(TRIM(input), 5, 1) = '-' THEN
                            CAST(SUBSTRING(TRIM(input), 7) AS INTEGER)
                        -- Without dash: '2022Q3' -> 3, '2022M06' -> 6
                        ELSE CAST(SUBSTRING(TRIM(input), 6) AS INTEGER)
                    END AS number
            )
            SELECT {
                'start_date': CASE parsed.indicator
                    WHEN 'A' THEN MAKE_DATE(parsed.year, 1, 1)
                    WHEN 'S' THEN MAKE_DATE(parsed.year, (parsed.number - 1) * 6 + 1, 1)
                    WHEN 'Q' THEN MAKE_DATE(parsed.year, (parsed.number - 1) * 3 + 1, 1)
                    WHEN 'M' THEN MAKE_DATE(parsed.year, parsed.number, 1)
                    WHEN 'W' THEN CAST(
                        STRPTIME(parsed.year || '-W' || LPAD(CAST(parsed.number AS VARCHAR), 2, '0') || '-1', '%G-W%V-%u')
                        AS DATE
                    )
                    WHEN 'D' THEN CAST(
                        STRPTIME(parsed.year || '-' || LPAD(CAST(parsed.number AS VARCHAR), 3, '0'), '%Y-%j')
                        AS DATE
                    )
                END,
                'end_date': CASE parsed.indicator
                    WHEN 'A' THEN MAKE_DATE(parsed.year, 12, 31)
                    WHEN 'S' THEN LAST_DAY(MAKE_DATE(parsed.year, parsed.number * 6, 1))
                    WHEN 'Q' THEN LAST_DAY(MAKE_DATE(parsed.year, parsed.number * 3, 1))
                    WHEN 'M' THEN LAST_DAY(MAKE_DATE(parsed.year, parsed.number, 1))
                    WHEN 'W' THEN CAST(
                        STRPTIME(parsed.year || '-W' || LPAD(CAST(parsed.number AS VARCHAR), 2, '0') || '-7', '%G-W%V-%u')
                        AS DATE
                    )
                    WHEN 'D' THEN CAST(
                        STRPTIME(parsed.year || '-' || LPAD(CAST(parsed.number AS VARCHAR), 3, '0'), '%Y-%j')
                        AS DATE
                    )
                END,
                'period_indicator': parsed.indicator
            }::vtl_time_period
            FROM parsed
        )
    END
);
