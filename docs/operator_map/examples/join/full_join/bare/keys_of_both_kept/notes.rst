:Tests: the full join of two operands with the same identifiers, with no clause.
:Manual: `Full Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html>`__: when all the operands have the same identifiers, "the using clause is optional"; the join is "according to SQL full outer join", and for a key that is not found "the Measures take the null value of their respective domains".
:Expected: the five keys of both operands, with ``Me_2`` null at ``(1,B)`` (in ``DS_1`` only) and ``Me_1`` null at ``(1,C)`` (in ``DS_2`` only).
