:Tests: a scalar operand with ``numDigit`` given by an Integer component, inside ``calc``.
:Manual: `Rounding <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__ types ``numDigit`` as ``component<integer> | integer``, so it may be a component while the operand is a scalar.
:Expected: the error ``1-1-15-8``, "At op {op}: {op} operator cannot have a {comp_type} as parameter": by a team decision, ``numDigit`` is a Component only when the operand is a Component too.
