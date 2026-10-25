:Tests: ``DS_1 [ calc Me_2 := Me_1 and true ]``: a new measure from a component and the boolean scalar true.
:Manual: `Logical conjunction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20conjunction/index.html>`__, Example 2, output verbatim: "The **and** operator returns **true** if both operands are **true**, otherwise **false**."
:Expected: the page's six data points: ``Me_2`` equals ``Me_1`` on each one, as ``and true`` leaves the other operand as it is.
