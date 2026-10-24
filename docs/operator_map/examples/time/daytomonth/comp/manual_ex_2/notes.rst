:Tests: ``daytomonth(Me_1)`` inside ``calc``: numbers of days to months and days.
:Manual: `Number days to duration <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Number%20days%20to%20duration/index.html>`__, Example 2: "The operator **daytomonth** returns a duration having the following mask", ``\PM\MDD\D``, with the months and days "according to ISO 8601 M = 30D".
:Expected: The manual's durations, ``P8M0D``, ``P24M4D`` and ``P35M6D``, a Duration ``Me_2``: 8 × 30, 24 × 30 + 4 and 35 × 30 + 6 days.
:Finding: Both engines give ``Me_2`` the type String, not Duration.
:Erratum: As in the page's Example 1, of ``daytoyear``, the manual's output repeats ``Id_1 = G`` on its second data point, where ``DS_1`` has ``H``; the expected output has ``H``.
