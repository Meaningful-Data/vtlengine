Authored. `nvl` of two components of the same Data Set inside `calc`.

Derivation (Nvl page: returns op2 when op1 is null, otherwise op1; a null op2 gives null):

- Id_1 = 1: Me_1 = 10 is not null -> Me_3 = 10.
- Id_1 = 2: Me_1 is null -> Me_3 = Me_2 = 200.
- Id_1 = 3: Me_1 is null -> Me_3 = Me_2, which is null as well -> Me_3 is null.
