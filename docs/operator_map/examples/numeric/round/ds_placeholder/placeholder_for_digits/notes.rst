:Tests: the placeholder ``_`` for ``numDigit``, on a dataset with one Number measure.
:Manual: `Rounding <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__: "``round ( 12345.6, _ )`` gives ``12346``"; the result is typed ``dataset { measure<number> _+ }``, so the measure keeps its name.
:Expected: the same as ``round(DS_1)``: ``Me_1`` keeps its name and becomes an Integer, 3, 3 and 12346.
:Finding: both engines return the measure as ``int_var``, not ``Me_1``.
