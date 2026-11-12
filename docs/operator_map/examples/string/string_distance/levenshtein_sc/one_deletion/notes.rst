:Tests: ``string_distance(levenshtein, SC_1, SC_2)`` with ``SC_1`` = ``foo`` and ``SC_2`` = ``fo``, the Levenshtein distance of two scalars.
:Manual: `String distances <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20distances/index.html>`__: "Returns the Levenshtein distance, which is the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into another.", as in ``string_distance(levenshtein, "foo", "fo")`` gives ``1``.
:Expected: ``SC_r`` is 1, a Number: delete the last ``o``.
