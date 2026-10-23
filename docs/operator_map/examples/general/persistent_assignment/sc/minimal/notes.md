# Persistent assignment of a scalar

Authored from the Behaviour of [Persistent
assignment](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Persistent%20assignment/index.html): "If the Operand `op` is a scalar value, the
result Data Set has no Components and contains only such a scalar value. For example,
`income <- 3` assigns the value `3` to the persistent Data Set named income." The example names
the result `SC_r`, the map's name for a scalar result. The expected output is the scalar 3, an
integer, as for the non-persistent assignment `SC_r := 3`.

## The page contradicts itself

Its "Input parameters type" is `op :: dataset` and its "Result type" `result :: dataset`: by the
type signature, a scalar operand is not allowed. The Behaviour, quoted above, assigns one. The
example follows the Behaviour, which gives the result, and the non-persistent assignment, whose
types do allow a scalar (`op :: dataset | scalar`).
