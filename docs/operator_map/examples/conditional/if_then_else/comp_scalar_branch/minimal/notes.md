Authored. Component-level `if` inside `calc` where the branches are scalars, the usual way to
build a flag. The page (l. 92-93): "If one of the operands thenOperand or elseOperand is scalar,
such a scalar value can be returned depending on the outcome of the condition." Here both are.

Derivation (if-then-else `content.rst`, Behaviour at Component level l. 90-93):

- Id_1 = 1: 5 > 10 is false -> elseOperand 0.
- Id_1 = 2: 15 > 10 is true -> thenOperand 1.
- Id_1 = 3: 10 > 10 is false -> elseOperand 0.
