:Tests: ``timeshift(DS_2, 2)`` on annual date series, where "conventionally each period is identified by its last day".
:Manual: `Time shift <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20shift/index.html>`__, Example 2, output verbatim: the operator "shifts the reference time Identifier of a number of periods (of the time series) equal to the *shiftNumber* parameter".
:Expected: Each date moves two years on, 2010-12-31 to 2013-12-31 becoming 2012-12-31 to 2015-12-31, with the measures as they were.
