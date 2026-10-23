Authored. `nvl` with a scalar as op1 and a component as op2 inside `calc`. The page allows a
scalar, a component or a Data Set in both positions (Nvl `content.rst` l. 37-47), and the typical
behaviour page applies the operator "on a Component and a Scalar value".

Derivation (Nvl `content.rst` l. 71: returns op2 when op1 is null, otherwise op1): op1 is 5, which
is not null, so Me_2 = 5 on every data point, also on Id_1 = 2 where Me_1 is null. Me_1 is
untouched.
