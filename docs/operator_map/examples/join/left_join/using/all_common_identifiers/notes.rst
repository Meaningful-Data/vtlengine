:Tests: ``using Id_1, Id_2`` on operands with the same identifiers.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: "When present, the using clause specifies, for each join operand except the left-most one, at least one of its identifiers in common with the left-most operand, and an additional nvl() rule for all its other identifiers".
:Expected: both identifiers of ``DS_2`` are named, so no ``nvl()`` is needed and the keys are those without ``using``: the result of Left keys kept.
