:Tests: ``yeartoday(cast("P1Y20D", duration))``, the page's Example 2, the duration cast from a string, as VTL has no duration literal.
:Manual: `Duration to number days <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html>`__: ``yeartoday (P1Y20D);`` "returns 385", "according to ISO 8601 Y = 365D"; and the manual's default conversion (Type conversion) reads the string: "If the input string starts with ``P``, the entire string is consumed and parsed, according to the ISO-8601 standard for durations up to a precision of days".
:Expected: ``DS_r = 385``, an Integer: 365 + 20.
:Finding: Both engines refuse the cast with SemanticError ``1-1-5-6``: "Impossible to cast P1Y20D from type String to Duration".
