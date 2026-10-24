:Tests: ``substr(DS_1, 1, 5)``: the first five characters of every value of both measures.
:Manual: `Sub-string extraction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Sub-string%20extraction/index.html>`__, Example 2, output verbatim: "The substring starts from the startth character of the input string and has a number of characters equal to the ``length`` parameter."
:Expected: ``Me_1`` is ``hello``, ``abcde`` and ``pqrst``; ``Me_2`` is ``mediu``, ``short`` and ``"this "`` (with its trailing space).
