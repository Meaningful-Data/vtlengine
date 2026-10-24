:Tests: ``monthtoday(Me_1)`` inside ``calc``, on durations written as the manual writes them (``P8M0D``, ``P24M4D``...): vtlengine's Duration holds only a period indicator, so it has no representation of them.
:Manual: `Duration to number days <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html>`__: ``monthDuration`` is "a duration having the following mask", ``\PM\MDD\D``, and the days are counted "according to ISO 8601 M = 30D", as in Example 3, where ``monthtoday (P3M10D);`` "returns 100".
:Expected: ``Me_2`` is 65 (2 × 30 + 5), 360 (12 × 30) and 15, Integers.
:Finding: Both engines refuse to load the durations, with DataLoadError ``0-3-1-6``.
