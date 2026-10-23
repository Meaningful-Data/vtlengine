`string_distance(levenshtein, "kitten", DS_1)`: a scalar and a dataset, the other order of
`levenshtein_ds_sc`. The scalar is composed with the only string measure of `DS_1`. "All distance
methods are symmetric (commutative)", so the order of the operands does not change the distance.

- `"kitten"` to `"sitting"`: 3 (substitute `k` by `s`, substitute `e` by `i`, insert `g`).
- `"kitten"` to `"kitten"`: 0.
- `"kitten"` to `"mitten"`: 1 (substitute `k` by `m`).
- Row 4 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.

The result measure is `num_var`: `string_distance` changes the data type (string to number), so
the typical behaviour of the operators changing the data type (Typical behaviours of the ML
Operators, "Operators changing the data type") replaces the operand measure by the default
measure of the result type, `num_var` for number. The page's Result type,
`dataset { measure<number> _+ }`, names no measure, and its examples 1 and 2 give `num_var`.
