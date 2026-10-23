`string_distance(levenshtein, DS_1, "fo")`: a dataset and a scalar. The page types `string1` and
`string2` as dataset, component or string, and its behaviour is that of the operators applicable
on two datasets, components or scalars: "An operand Scalar value is composed with all the Measures
of the other operand" (Typical behaviours), here the only string measure of `DS_1` (the page
requires exactly one).

- `"foo"` to `"fo"`: 1 (delete one `o`).
- `"bar"` to `"fo"`: 3 (no character in common, so two substitutions and one deletion).
- `"fo"` to `"fo"`: 0.
- Row 4 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.

The result measure is `num_var`: `string_distance` changes the data type (string to number), so
the typical behaviour of the operators changing the data type (Typical behaviours of the ML
Operators, "Operators changing the data type") replaces the operand measure by the default
measure of the result type, `num_var` for number. The page's Result type,
`dataset { measure<number> _+ }`, names no measure, and its examples 1 and 2 give `num_var`.
