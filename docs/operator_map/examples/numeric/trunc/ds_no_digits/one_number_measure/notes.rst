:Tests: ``trunc`` with ``numDigit`` omitted, on a dataset with one Number measure.
:Manual: `Truncation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Truncation/index.html>`__: "The basic scalar type of the result is *integer* if *numDigit* is omitted"; the result is typed ``dataset { measure<number> _+ }``, and the unary operators give "homonymous Measures in DS_r".
:Expected: ``Me_1`` keeps its name and becomes an Integer: 3.14159, 2.71828 and 12345.6 give 3, 2 and 12345. Ceiling and Floor Example 1 do the same, and nothing puts ``trunc`` among the "Operators changing the data type", which would rename the measure ``int_var``.
:Finding: both engines return the measure as ``int_var``, not ``Me_1``.
