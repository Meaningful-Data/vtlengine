`instr(DS_1, "o", 2, _)`: the trailing `_` stands for the omitted `occurrence`, so this is
`instr(DS_1, "o", 2)` with the underscore written out.
The manual's syntax conventions say that "a non-specified optional operand is substituted by
an underscore or, if it is in the end of the invocation, can be omitted" (Overview and
conventions, syntax conventions).
With `occurrence` omitted its value is 1 ("If nth occurrence is omitted, the value is 1"), and the
search starts from the 2nd character. At dataset level the result is a single measure `int_var`
of type integer.

- `"hello world"` (h=1, e=2, l=3, l=4, o=5): the first `o` from the 2nd character on is the one at 5.
- `"o-o"` (`o` at 1 and 3): the `o` at 1 is before the 2nd character, so the result is 3.
- `"abc"`: there is no occurrence, so the result is 0.
- Row 4 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
