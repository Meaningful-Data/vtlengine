:Tests: ``ratio_to_report`` in ``calc``, partitioned by ``Id_1``.
:Manual: `Ratio to report <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Ratio%20to%20report/index.html>`__: "The operator returns the ratio between the value of the current Data Point and the sum of the values of the partition which the current Data Point belongs to."
:Expected: the sums of ``Me_1`` are 10 in ``A`` and 12 in ``B``: ``Me_3`` is 0.3, 0.1, 0.5, 0.1 in ``A``, and 4 / 12 = 0.333333 and 8 / 12 = 0.666667 in ``B``.
