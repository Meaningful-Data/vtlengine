Manual: [Duration to number days](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html): `monthtoday` takes "a duration having the following
mask: `\PM\MDD\D`" and returns the number of days, "according to ISO 8601 M = 30D" (the page's
Example 3, `monthtoday (P3M10D)` returns 100, follows the same rule):

- P2M5D = 2 × 30 + 5 = 65;
- P12M0D = 12 × 30 + 0 = 360;
- P0M15D = 0 × 30 + 15 = 15.

The durations are written as the manual writes them (`P8M0D`, `P24M4D`...). vtlengine's Duration
holds only a period indicator, so it has no representation of them: a failure to load them is
an engine finding.
