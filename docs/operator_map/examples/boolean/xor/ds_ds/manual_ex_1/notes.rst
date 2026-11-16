:Tests: ``DS_1 xor DS_2`` on two data sets with the same identifiers and one boolean measure.
:Manual: `Exclusive disjunction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Exclusive%20disjunction/index.html>`__, Example 1, output verbatim: "The **xor** operator returns **true** if only one of the operand is **true** (but not both), **false** otherwise."
:Expected: the page's six data points: ``Me_1`` is true where exactly one operand is true, and false for (M, 65, B, 2013), where both are true, and (F, 64, U, 2013), where both are false.
