`string_distance(levenshtein, comp, comp)` inside `calc`: `Me_3` is a new measure of type number
holding the Levenshtein distance, "the minimum number of single-character edits (insertions,
deletions, or substitutions) required to change one string into another" (String distances).

- `"kitten"` to `"sitting"`: 3 (substitute `k` by `s`, substitute `e` by `i`, insert `g`).
- `"flaw"` to `"lawn"`: 2 (delete `f`, insert `n` at the end), checked with the full table of
  the dynamic programme.
- `"same"` to `"same"`: 0.
- Row 4 has a null `Me_2`. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
