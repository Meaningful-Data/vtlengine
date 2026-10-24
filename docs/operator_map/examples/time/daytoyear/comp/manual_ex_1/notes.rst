:Tests: ``daytoyear(Me_1)`` inside ``calc``: numbers of days to years and days.
:Manual: `Number days to duration <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Number%20days%20to%20duration/index.html>`__, Example 1: "The operator **daytoyear** returns a duration having the following mask", ``\PY\YDDD\D``, with the years and days "according to ISO 8601 Y = 365D".
:Expected: The manual's durations, ``P0Y240D``, ``P1Y359D`` and ``P2Y326D``, a Duration ``Me_2``: 240, 365 + 359 and 2 × 365 + 326 days.
:Finding: Both engines give ``Me_2`` the type String, not Duration.
:Erratum: The manual's output repeats ``Id_1 = G`` on its second data point, where ``DS_1`` has ``H``: the ``calc`` keeps the identifiers of ``DS_1``, and two data points cannot share the key ``G``. The expected output has ``H``.
