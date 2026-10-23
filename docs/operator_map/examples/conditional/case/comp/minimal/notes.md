Authored. Component-level `case` with one `when` inside `calc`, branches that are components.

Derivation (Case page, Behaviour at Component level: for each data point, the value of the
thenOperand of the first `when` whose condition is true, else the value of elseOperand):

- Id_1 = 1: 10 < 5 is false -> elseOperand Me_2 = 5.
- Id_1 = 2: 3 < 8 is true -> thenOperand Me_1 = 3.
- Id_1 = 3: 7 < 7 is false -> elseOperand Me_2 = 7.
