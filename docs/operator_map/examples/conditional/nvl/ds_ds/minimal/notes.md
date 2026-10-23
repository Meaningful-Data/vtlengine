Authored. `nvl` on two Data Sets with the same components and partly different identifiers.

Derivation (Nvl `content.rst` l. 71-73: "the typical behaviour of the operators applicable on two
scalar values or Data Sets or Data Set Components"; typical behaviour page: the operand Data Sets
are joined on the common identifiers with an inner join, "Data Points that are not coupled are
left out"; then, for each couple of data points, nvl returns the value of op2 when the value of
op1 is null, otherwise the value of op1):

- Id_1 = 1 is only in DS_1 and Id_1 = 4 only in DS_2: both are left out.
- Id_1 = 2: DS_1 is null -> the value of DS_2, 200.
- Id_1 = 3: DS_1 is 30, not null -> 30 (DS_2's 300 is not used).
