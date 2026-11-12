:Tests: ``daytomonth(SC_1)`` with ``SC_1`` = 134, the page's Example 3.
:Manual: `Number days to duration <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Number%20days%20to%20duration/index.html>`__: ``daytomonth (134);`` "returns P4M14D"; **daytomonth** "returns a duration having the following mask", ``\PM\MDD\D``, "the corresponding number of months and days; according to ISO 8601 M = 30D".
:Expected: ``DS_r = P4M14D``, a String: 134 = 4 × 30 + 14.
:Team decision: `#394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__: a Duration holds only a period indicator (``A``, ``S``, ``Q``, ``M``, ``W``, ``D``), so the ISO 8601 durations that ``daytoyear`` and ``daytomonth`` return and ``yeartoday`` and ``monthtoday`` take are Strings.
