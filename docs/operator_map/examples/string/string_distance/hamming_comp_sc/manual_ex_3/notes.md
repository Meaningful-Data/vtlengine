The manual's example 3 of String distances, imported as it is. It runs `hamming` on a component and
a scalar inside `calc`, which is why this variant exists next to `hamming_sc`: the manual's example
is not a scalar one.

`"foo"` and `"fob"` differ at the 3rd position only (1); `"bar"` and `"fob"` differ at all three (3).
