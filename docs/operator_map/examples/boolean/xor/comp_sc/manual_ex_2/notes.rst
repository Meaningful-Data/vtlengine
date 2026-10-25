:Tests: ``DS_1 [ calc Me_2 := Me_1 xor true ]``: a new measure from a component and the boolean scalar true.
:Manual: `Exclusive disjunction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Exclusive%20disjunction/index.html>`__, Example 2, output verbatim: "The **xor** operator returns **true** if only one of the operand is **true** (but not both), **false** otherwise."
:Expected: the page's six data points: ``Me_2`` is the negation of ``Me_1`` on each one, as ``xor true`` flips the operand.
