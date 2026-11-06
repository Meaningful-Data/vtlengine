:Tests: ``monthtoday(SC_1)`` with ``SC_1`` = ``P3M10D``, a String, the page's Example 3.
:Manual: `Duration to number days <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html>`__: ``monthtoday (P3M10D);`` "returns 100", "according to ISO 8601 M = 30D".
:Expected: ``DS_r = 100``, an Integer: 3 × 30 + 10.
:Team decision: `#394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__: a Duration holds only a period indicator (``A``, ``S``, ``Q``, ``M``, ``W``, ``D``), so the ISO 8601 durations that ``daytoyear`` and ``daytomonth`` return and ``yeartoday`` and ``monthtoday`` take are Strings.
