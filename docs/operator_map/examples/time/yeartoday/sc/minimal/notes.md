Manual: [Duration to number days](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html), Example 2 (in the text of the page):
`yeartoday (P1Y20D);` "returns 385": 1 × 365 + 20 ("according to ISO 8601 Y = 365D").

VTL has no duration literal: the operand is the cast of a string, which the manual's default
conversion to duration reads as an ISO 8601 duration ("If the input string starts with P, the
entire string is consumed and parsed, according to the ISO-8601 standard for durations up to a
precision of days", Type conversion).
