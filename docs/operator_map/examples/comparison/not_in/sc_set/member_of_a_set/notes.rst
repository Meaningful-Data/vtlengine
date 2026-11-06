:Tests: ``SC_1 not_in { "b", "hello", "c" }`` with ``SC_1`` = ``"b"``: a scalar that is an element of a set literal.
:Manual: `Element of <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Element%20of/index.html>`__, semantics for scalar operations: "``"b" not_in { "b", "hello", "c" }`` returns ``false``".
:Expected: ``SC_r = false``, a Boolean: b is an element of the set, so ``not_in`` is false.
