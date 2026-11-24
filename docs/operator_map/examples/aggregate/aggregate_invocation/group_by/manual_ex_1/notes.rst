:Tests: ``avg`` at data set level grouped by ``Id_1``, on an operand with the non-viral attribute ``At_1``.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__, Example 1, output verbatim: "For invocation at Data Set level, the Attribute propagation rule is applied".
:Expected: ``At_1`` is dropped, as a non-viral attribute; 2010 gives (20 + 1 + 1 + 23) / 4 = 11.25, 2011 gives 10 and 2012 gives 35.
:Erratum: the script comments its two alternative forms with ``#``, which is the membership operator, not a VTL comment. The manual's conventions define ``/* */`` and ``//``, so they are commented with ``//`` here.
