:Tests: ``DS_1[calc Me_3 := Me_1 xor Me_2]``: ids 1 to 4 hold the four combinations of two boolean values.
:Manual: `Exclusive disjunction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Exclusive%20disjunction/index.html>`__: "The **xor** operator returns **true** if only one of the operand is **true** (but not both), **false** otherwise."
:Expected: ``Me_3`` is true for ids 2 and 3, where exactly one operand is true, and false for ids 1 and 4.
