:Tests: the natural inner join of two operands with the same identifiers, with no clause.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: "if using is not specified, inner_join performs a natural join, that is, it uses all the identifiers appearing in more than one dataset as join keys", and matches them "according to SQL inner join".
:Expected: the keys in both operands, ``(1,A)``, ``(2,A)`` and ``(2,B)``, with ``Me_1`` and ``Me_2``; ``(1,B)`` is in ``DS_1`` only and ``(1,C)`` in ``DS_2`` only.
