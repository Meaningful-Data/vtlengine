The manual's example (Sub-string extraction): `substr("abcdefghijklmnopqrstuvwxyz", 30, 10)` gives
`""`: "If `start` is greater than the length of the input string, an empty string is extracted".

The comparison of the map reads an empty value and a null as the same one, so this example
cannot tell an empty string from a null.
