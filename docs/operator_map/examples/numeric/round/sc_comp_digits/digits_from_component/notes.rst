:Tests: a scalar operand with ``numDigit`` given by an Integer component, inside ``calc``.
:Manual: `Rounding <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__ types ``numDigit`` as ``component<integer> | integer``, so it may be a component while the operand is a scalar.
:Expected: ``Me_2 = round(3.14159, Me_1)`` is 3.14 for 2 digits, 3.1416 for 4 and 3.0 for 0, a Number; the other components are kept.
:Finding: both engines refuse with SemanticError ``1-1-15-8``: "round operator cannot have a DataComponent and an Scalar operand as parameter".
