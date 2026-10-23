Authored. Component-level `if` inside `calc`: the page's own valid syntax `if A > B then A else B`
with A = Me_1 and B = Me_2.

Derivation (if-then-else page, Behaviour at Component level: the operation is applied to each
data point of the input Data Set; the new component Me_3 is added as a measure):

- Id_1 = 1: 10 > 5 is true -> Me_1 = 10.
- Id_1 = 2: 3 > 8 is false -> Me_2 = 8.
- Id_1 = 3: 7 > 7 is false -> Me_2 = 7.
