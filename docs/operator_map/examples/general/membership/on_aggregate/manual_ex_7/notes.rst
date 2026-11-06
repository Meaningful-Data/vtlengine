:Tests: ``max(DS_1)#Me_1``: membership on the result of an aggregate invocation, which has no identifiers, gives a scalar.
:Manual: `Membership <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html>`__, Example 7, output verbatim: "If ds has no Identifier Components (for example is a result of an aggregate invocation), [...] the operator returns a scalar value equal to the value of ``comp`` in ``ds``".
:Expected: ``max(DS_1)`` has one data point, ``Me_1 = 3`` and ``Me_2 = 12``, so ``SC_r = 3``, an Integer.
