The manual's example 1 of String distances, imported as it is. At dataset level the result has one
measure of type number, which the expected output of the example calls `num_var`.

The result measure is `num_var`: `string_distance` changes the data type (string to number), so the
typical behaviour of the operators changing the data type (Typical behaviours of the ML Operators,
"Operators changing the data type") replaces the operand measure by the default measure of the
result type, `num_var` for number. The page's Result type, `dataset { measure<number> _+ }`, names
no measure, and its examples 1 and 2 give `num_var`.

`"foo"` to `"fo"` is 1 (delete one `o`) and `"bar"` to `"baz"` is 1 (substitute `r` by `z`).
