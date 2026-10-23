`instr` with `pattern`, `start` and `occurrence` all taken from components. The manual types
`occurrence` as `component<integer [value >= 1]>` as well as an integer (String pattern location,
Input parameters type). Each data point uses its own pattern (`Me_2`), start (`Me_3`) and
occurrence (`Me_4`), and `calc` stores the position in the new measure `Me_5`.

- `"hello world"`, pattern `"o"`, start 1, occurrence 2: the `o` are at 5 and 8, so the 2nd one is
  at 8.
- `"banana"`, pattern `"an"`, start 3, occurrence 1: `"an"` begins at 2 and at 4, and the first one
  from the 3rd character on is the one at 4.
- `"o-o-o"`, pattern `"o"`, start 1, occurrence 3: the `o` are at 1, 3 and 5, so the 3rd one is at
  5.
- `"hello"`, pattern `"o"`, start 1, occurrence 2: the only `o` is at 5, so there is no 2nd
  occurrence and the result is 0 ("the returned value is 0").
