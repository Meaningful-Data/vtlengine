`damerau_levenshtein` "extends Levenshtein distance by including transpositions of adjacent
characters as a single operation" (String distances). `"form"` and `"from"` differ only by
the adjacent `or` / `ro`, so the distance is 1 (with plain Levenshtein it would be 2, two
substitutions). The result type is number.
