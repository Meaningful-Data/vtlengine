:Tests: ``filter Me_1 > 10``, where ``Me_1`` is null on one data point, so the condition is null there.
:Manual: `Filtering Data Points <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Filtering%20Data%20Points/index.html>`__: "it filters out the Data Points of the operand Data Set for which filterCondition condition evaluates to false or null".
:Expected: ``Id_1`` = 3 and 4 (15 and 20) are kept; ``Id_1`` = 1 (5, false) and ``Id_1`` = 2 (null) are filtered out.
