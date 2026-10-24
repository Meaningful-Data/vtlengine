:Tests: ``daytoyear(782)``, the page's Example 2.
:Manual: `Number days to duration <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Number%20days%20to%20duration/index.html>`__: ``daytoyear (782);`` "returns P2Y52D"; **daytoyear** "returns a duration having the following mask", ``\PY\YDDD\D``, "the corresponding number of years and days; according to ISO 8601 Y = 365D".
:Expected: ``DS_r = P2Y52D``, a Duration: 782 = 2 × 365 + 52.
:Finding: Both engines give the result the type String, not Duration.
