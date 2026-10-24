:Tests: ``monthtoday(cast("P3M10D", duration))``, the page's Example 3, the duration cast from a string, as for ``yeartoday``.
:Manual: `Duration to number days <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html>`__: ``monthtoday (P3M10D);`` "returns 100", "according to ISO 8601 M = 30D".
:Expected: ``DS_r = 100``, an Integer: 3 × 30 + 10.
:Finding: Both engines refuse the cast with SemanticError ``1-1-5-6``: "Impossible to cast P3M10D from type String to Duration".
