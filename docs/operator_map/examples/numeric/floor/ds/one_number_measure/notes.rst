:Tests: ``floor`` on a dataset with one Number measure, which becomes an Integer measure.
:Manual: `Floor <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Floor/index.html>`__ types the result ``dataset { measure<integer> _+ }``, and its Example 1 turns two Number measures into Integer ones with the same names. Only the pages that rename say so, such as String length and String pattern location: ``dataset { measure<integer[value >= 0]> int_var }``.
:Expected: ``Me_1`` keeps its name: 7.2, -3.7 and 5.0 give 7, -4 and 5. The section "Operators changing the data type" would rename it ``int_var``, but nothing puts ``floor`` there.
:Finding: both engines return the measure as ``int_var``, not ``Me_1``.
