:Tests: ``ceil`` on a dataset with one Number measure, which becomes an Integer measure.
:Manual: `Ceiling <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Ceiling/index.html>`__ types the result ``dataset { measure<integer> _+ }``, and its Example 1 turns two Number measures into Integer ones with the same names. Only the pages that rename say so, such as String length and String pattern location: ``dataset { measure<integer[value >= 0]> int_var }``.
:Expected: ``Me_1`` keeps its name: 7.2, -3.7 and 5.0 give 8, -3 and 5. The section "Operators changing the data type" would rename it ``int_var``, but nothing puts ``ceil`` there.
:Finding: both engines return the measure as ``int_var``, not ``Me_1``.
