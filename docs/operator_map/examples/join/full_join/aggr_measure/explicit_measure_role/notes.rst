:Tests: ``aggr`` with the role keyword ``measure`` written out.
:Manual: `Full Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html>`__: ``aggrRole ::= { measure | attribute | viral attribute }``, and "if the role is not specified, it will be measure".
:Expected: the result of Sum by id 1, which leaves the role out: ``Me_3`` is 30 for ``Id_1`` = 1 and 70 for ``Id_1`` = 2.
