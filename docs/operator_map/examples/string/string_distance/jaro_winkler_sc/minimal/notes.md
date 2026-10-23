`string_distance(jaro_winkler, "FOO", "FOO")`: two identical strings.

The manual (String distances) says that `jaro_winkler` "Returns the Jaro–Winkler distance" and links
the Wikipedia article of that name, which defines the distance from the similarity: "The score is
normalized such that 0 means an exact match and 1 means there is no similarity. The original paper
actually defined the metric in terms of similarity, so the distance is defined as the inversion of
that value (distance = 1 − similarity)" (article read on 2026-10-23). `string_distance` is a
distance operator, as the other three methods are: "Returns the distance between two strings".

Two identical strings have a Jaro similarity of 1 (every character matches and none is
transposed), and the Winkler prefix boost cannot raise it above 1, so the Jaro-Winkler similarity
is 1 and the distance is exactly 0. The result type is number.

The manual gives no value for this operator beyond "a value between `0` and `1`" for
`string_distance(jaro_winkler, "FOO", "BAR")`. The identical strings of this example and the fully
disjoint ones of `minimal_2` are the two cases whose distance needs no approximation.
