The manual's example 2 of String distances, imported as it is. It runs `damerau_levenshtein` on two
datasets, which is why this variant exists next to `damerau_levenshtein_sc`: the manual's example
is not a scalar one. At dataset level the result has one measure of type number, which the expected
output of the example calls `num_var`.

`"foo"` to `"fo"` is 1 (delete one `o`) and `"bar"` to `"baz"` is 1 (substitute `r` by `z`), the
same as for Levenshtein: there is no adjacent transposition in these pairs.
