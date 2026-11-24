:Tests: ``string_distance(damerau_levenshtein, SC_1, SC_2)`` with ``SC_1`` = ``form`` and ``SC_2`` = ``from``, two scalars that differ by an adjacent transposition.
:Manual: `String distances <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20distances/index.html>`__: "Returns the Damerau–Levenshtein distance, which extends Levenshtein distance by including transpositions of adjacent characters as a single operation."
:Expected: ``SC_r`` is 1, a Number: ``or`` and ``ro`` differ by one transposition, where Levenshtein would count two substitutions.
