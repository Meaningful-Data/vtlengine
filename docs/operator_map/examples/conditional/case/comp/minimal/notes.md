Authored. Component-level `case` with one `when` inside `calc`, branches that are components.

Derivation (Case `content.rst`, Behaviour at Component level l. 85-89: the operation is applied
for each data point, and the result is the value of the thenOperand whose condition is true, or
the value of elseOperand when none is). With a single `when` the conditions cannot overlap, so
the "first" of l. 31 does not come into play here (`comp_multi_when` has overlapping `when`s):

- Id_1 = 1: 10 < 5 is false -> elseOperand Me_2 = 5.
- Id_1 = 2: 3 < 8 is true -> thenOperand Me_1 = 3.
- Id_1 = 3: 7 < 7 is false -> elseOperand Me_2 = 7.
