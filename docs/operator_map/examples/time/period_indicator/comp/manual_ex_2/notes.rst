:Tests: ``period_indicator(Id_3)`` inside ``filter``, compared with ``cast("A", duration)``: the annual data points are kept.
:Manual: `Period indicator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Period%20indicator/index.html>`__, Example 2, output verbatim: "The period indicator is the part of the *time_period* value which denotes the duration of the time period (e.g. day, week, month...)."
:Expected: ``2010`` has the indicator ``A`` and ``2013Q1`` has ``Q``, so the data point of ``2010`` is kept, with all its components.
