:Tests: ``cast(Me_1, time_period, "YYYY\QQ")`` on a String measure: the mask of the page's Example 9, ``cast ( "2000Q1", time_period, "YYYY\QQ" )``, "The first quarter of year 2000".
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "The special characters and the special words, if prefixed by the reverse slash [...] in the mask, appear in the same position in the time format", so ``\Q`` is the letter ``Q``, and ``Q`` the quarter.
:Expected: ``2000Q1``, the first quarter of 2000, and ``2001Q4``.
:Finding: Both engines raise SemanticError ``1-3-13``: "Unsupported VTL construct: cast with the mask ``"YYYY\QQ"``".
