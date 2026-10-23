`instr(comp, comp, comp)`: `pattern` and `start` are components, not constants. The manual types
`pattern` as `component<string>` and `start` as `component<integer [value >= 1]>` as well as a
string and an integer (String pattern location, Input parameters type). `occurrence` is left out,
so it is 1. Each data point uses its own pattern (`Me_2`) and start (`Me_3`), and the result is the
position in the whole string `Me_1`, stored by `calc` in the new measure `Me_4`.

- `"hello world"`, pattern `"o"`, start 6: the `o` at 5 is before the 6th character, so the first
  one found is the `o` at 8.
- `"banana"`, pattern `"an"`, start 3 (b=1, a=2, n=3, a=4, n=5, a=6): `"an"` begins at 2 and at 4,
  and the first one from the 3rd character on is the one at 4.
- `"abc"`, pattern `"z"`, start 1: there is no occurrence, so the result is 0.
- Row 4 has a null `Me_1`. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.

No pattern begins right at its start position, which would leave open whether the search is
inclusive of `start`.
