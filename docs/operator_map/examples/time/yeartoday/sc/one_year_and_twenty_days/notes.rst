:Tests: ``yeartoday(SC_1)`` with ``SC_1`` = ``P1Y20D``, a String, the page's Example 2.
:Manual: `Duration to number days <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html>`__: ``yeartoday (P1Y20D);`` "returns 385", "according to ISO 8601 Y = 365D".
:Expected: ``DS_r = 385``, an Integer: 365 + 20.
:Team decision: `#394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__: a Duration holds only a period indicator (``A``, ``S``, ``Q``, ``M``, ``W``, ``D``), so the ISO 8601 durations that ``daytoyear`` and ``daytomonth`` return and ``yeartoday`` and ``monthtoday`` take are Strings.
