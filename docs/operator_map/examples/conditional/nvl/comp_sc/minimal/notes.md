Authored. `nvl` of a component and a scalar inside `calc`; the new measure Me_2 is added and Me_1
is untouched.

Derivation (Nvl page: returns op2 when op1 is null, otherwise op1):

- Id_1 = 1: Me_1 = 10 -> Me_2 = 10.
- Id_1 = 2: Me_1 is null -> Me_2 = 0 (Me_1 stays null).
- Id_1 = 3: Me_1 = 30 -> Me_2 = 30.
