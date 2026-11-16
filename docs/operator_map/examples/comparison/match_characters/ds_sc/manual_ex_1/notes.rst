:Tests: ``match_characters(DS_1, "[A-Za-z]{2}[0-9]{3}")``: the string measure of ``DS_1`` against a pattern.
:Manual: `Match characters <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Match%20characters/index.html>`__, Example 1, output verbatim: "**match_characters** returns **true** if *op* matches the regular expression *regexp*, **false** otherwise."
:Expected: ``bool_var`` is true for G (``AX123``) and false for R (``AX2J5``).
