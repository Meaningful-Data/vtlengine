:Tests: ``aggr`` with ``group except Id_2``, which groups by ``Id_1``.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__, and Aggregate invocation: with group except "the Data Points are grouped by the values of the Identifiers not specified in the clause. The specified Identifiers are dropped in the result".
:Expected: the left join keeps the four keys of ``DS_1``, with ``Me_2`` null at ``(1,B)``; the groups are those of ``group by Id_1``: ``Me_3`` is 30 for ``Id_1`` = 1 and 70 for ``Id_1`` = 2.
