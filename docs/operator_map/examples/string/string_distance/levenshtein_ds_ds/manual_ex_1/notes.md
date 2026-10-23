The manual's example 1 of String distances, imported as it is. At dataset level the result has one
measure of type number, which the expected output of the example calls `num_var`.

The manual does not settle that name. The page's Result type is `dataset { measure<number> _+ }`
and, for the operators applied on two datasets, the typical behaviour produces "homonymous Measures
in DS_r" (here `Me_1`), whereas the Behaviour paragraph says that for two datasets the result is "a
dataset with a single measure of type number" and the expected outputs of examples 1 and 2 name it
`num_var`. The import keeps the examples as the manual gives them, so `num_var` is expected.

`"foo"` to `"fo"` is 1 (delete one `o`) and `"bar"` to `"baz"` is 1 (substitute `r` by `z`).
