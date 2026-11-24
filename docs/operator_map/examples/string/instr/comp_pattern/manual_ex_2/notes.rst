:Tests: ``instr(Me_1, "hello")`` inside ``calc``, where the new measure ``Me_2`` takes the position.
:Manual: `String pattern location <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20pattern%20location/index.html>`__, Example 2, output verbatim: "The operator returns the position in the input string of a specified string ( ``pattern`` )."
:Expected: ``Me_2`` is 1, 5, 0 and 5; ``Me_1`` is kept.
