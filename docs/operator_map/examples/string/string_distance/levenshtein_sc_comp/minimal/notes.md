`string_distance(levenshtein, "fo", Me_1)` inside `calc`: a scalar and a component, in this order
(the manual's example 3, the `hamming_comp_sc` variant, has the component first). `Me_2` is a new
measure of type number and `Me_1` is kept.

- `"fo"` to `"foo"`: 1 (insert one `o`).
- `"fo"` to `"of"`: 2 (two substitutions; with Levenshtein the swap of two adjacent characters is
  not one operation, which only `damerau_levenshtein` allows).
- Row 3 has a null `Me_1`. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
