:Tests: ``sub Id_1 = 1, Id_2 = "B", Id_3 = "YY"``: every identifier fixed.
:Manual: `Subspace <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Subspace/index.html>`__, Example 2, output verbatim: "The resulting Data Set has the Identifier Components that are not specified as identifier (s) and has the same Measure and Attribute Components of the input Data Set."
:Expected: The data point ``(1,B,YY)`` alone, with no identifiers: ``Me_1`` = 9 and ``At_1`` = ``F``.
