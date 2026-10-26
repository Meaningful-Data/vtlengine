:Tests: ``using Id_1, Id_2`` on operands with the same identifiers.
:Manual: `Full Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html>`__: "The using clause is optional only if ``I_i = I_j`` ... otherwise it is mandatory, and it must specify a subset of the identifiers common to every join operand. For every other identifier not included in the using clause, a nvl() expression must be specified."
:Expected: both identifiers are named, so no ``nvl()`` is needed and the keys are those without ``using``: the result of Keys of both kept.
:Finding: both engines refuse any ``using`` clause on ``full_join`` with SemanticError ``1-1-13-8``, "No available using clause".
