:Tests: ``DS_1[calc Me_3 := Me_1 or Me_2]``: ids 1 to 4 hold the four combinations of two boolean values.
:Manual: `Logical disjunction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20disjunction/index.html>`__: "The **or** operator returns **true** if at least one of the operands is **true**, otherwise **false**."
:Expected: ``Me_3`` is false only for id 4, where both operands are false, and true for ids 1 to 3.
