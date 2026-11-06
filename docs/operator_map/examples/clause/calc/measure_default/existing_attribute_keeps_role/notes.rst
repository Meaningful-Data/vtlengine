:Tests: ``calc At_1 := "X"`` with no role: ``At_1`` is an Attribute of ``DS_1``.
:Manual: `Calculation of a Component <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Calculation%20of%20a%20Component/index.html>`__: "if the component exists in the operand Data Set then it maintains its role", and "If the Calculated component is a Measure or an Attribute that already exists in the input Data Set, the calculated values overwrite the original values".
:Expected: ``At_1`` stays an Attribute, with ``X`` on every data point; ``Id_1`` and ``Me_1`` are unchanged.
