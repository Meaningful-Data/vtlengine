`string_distance(levenshtein, "kitten", DS_1)`: a scalar and a dataset, the other order of
`levenshtein_ds_sc`. The scalar is composed with the only string measure of `DS_1`. "All distance
methods are symmetric (commutative)", so the order of the operands does not change the distance.

- `"kitten"` to `"sitting"`: 3 (substitute `k` by `s`, substitute `e` by `i`, insert `g`).
- `"kitten"` to `"kitten"`: 0.
- `"kitten"` to `"mitten"`: 1 (substitute `k` by `m`).
- Row 4 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.

The result measure is named `num_var`, as in the manual's examples 1 and 2 of String distances,
which are dataset examples. The page leaves the name open: its Result type is
`dataset { measure<number> _+ }` and the typical behaviour would give homonymous measures (here
`Me_1`), whereas its Behaviour paragraph says that for two datasets the result is "a dataset
with a single measure of type *number*".
