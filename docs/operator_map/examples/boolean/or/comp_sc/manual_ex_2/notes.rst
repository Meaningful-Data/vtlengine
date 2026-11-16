:Tests: ``DS_1 [ calc Me_2 := Me_1 or true ]``: a new measure from a component and the boolean scalar true.
:Manual: `Logical disjunction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20disjunction/index.html>`__, Example 2, output verbatim: "The **or** operator returns **true** if at least one of the operands is **true**, otherwise **false**."
:Expected: the page's six data points: ``Me_2`` is true on each one, as ``or true`` is true whatever the other operand is.
