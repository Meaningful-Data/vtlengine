:Tests: ``using Id_1, Id_2``, all the identifiers that the operands have in common.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: "if using is specified, inner_join removes from each of the ``I_i`` sets those identifiers not appearing in the using clause, and only uses the remaining identifiers as join keys if they appear in more than one dataset".
:Expected: the keys of the natural join, so the result of Common keys only: ``(1,A)``, ``(2,A)`` and ``(2,B)``.
