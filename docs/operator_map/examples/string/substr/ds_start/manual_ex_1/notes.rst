:Tests: ``substr(DS_1, 7)``: every value of both measures from the 7th character to its end.
:Manual: `Sub-string extraction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Sub-string%20extraction/index.html>`__, Example 1, output verbatim: "If ``length`` is omitted or overcomes the length of the input string, the substring ends at the end of the input string."
:Expected: ``Me_1`` is ``world``, ``ghilmno`` and ``vwxyz``; ``Me_2`` is ``" size text"`` (with its leading space), ``text`` and ``s a long description``.
