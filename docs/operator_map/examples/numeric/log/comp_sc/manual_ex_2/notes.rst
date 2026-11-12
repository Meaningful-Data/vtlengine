:Tests: ``log`` of a component to base 2, inside ``calc``, where the result replaces the measure ``Me_1``.
:Manual: `Logarithm <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Logarithm/index.html>`__, Example 2, output verbatim. As in Example 1, the first operand is the number and the second the base: "``log ( 1024, 2 )`` gives ``10``", although the parameter table says the opposite.
:Expected: ``Me_1 = log(Me_1, 2)`` gives 10.0, 6.0 and 5.0 for 1024, 64 and 32; ``Me_2`` is unchanged.
