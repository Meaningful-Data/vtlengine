Authored. `nvl` with a scalar as op1 and a Data Set as op2. The page allows a scalar, a component
or a Data Set in both positions (Nvl `content.rst` l. 37-47) and says that `nvl` "has the typical
behaviour of the operators applicable on two scalar values or Data Sets or Data Set Components"
(l. 73).

Derivation (typical behaviour page: "DS_1 or DS_2 may be replaced by a Scalar value", "An operand
Scalar value is treated as a Data Point that couples with all the Data Points of the other
operand" and "is composed with all the Measures of the other operand"; Nvl l. 71: the result is
the value of op2 when op1 is null, otherwise the value of op1):

- op1 is 5, which is not null, so the measure of every data point of the result is 5. That
  includes Id_1 = 2, where DS_1's Me_1 is null: op2 is not used.
- The identifiers are those of DS_1, and Me_1 is Integer.
