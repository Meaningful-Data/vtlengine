`string_distance(jaro_winkler, "FOO", "FOO")`: two identical strings.

The expected value rests on the page's "distance" wording plus the definition distance =
1 − similarity. The page lists `jaro_winkler` among "the distance method to apply", says that it
"Returns the Jaro–Winkler distance", and says that `string_distance` "Returns the distance between
two strings", so an exact match has to give a distance of 0. The page gives no formula. The usual
definition of the Jaro-Winkler distance is 1 minus the Jaro-Winkler similarity, which the Wikipedia
article that the page links says in its lead (that lead now names the metric the "Jaro–Winkler
similarity"): "The score is normalized such that 0 means an exact match and 1 means there is no
similarity. The original paper actually defined the metric in terms of similarity, so the distance
is defined as the inversion of that value (distance = 1 − similarity)" (article read on
2026-10-23).

Two identical strings have a Jaro similarity of 1 (every character matches and none is
transposed), and the Winkler prefix boost cannot raise it above 1, so the Jaro-Winkler similarity
is 1 and the distance is exactly 0. The result type is number.

The page gives no value for this operator beyond "a value between `0` and `1`" for
`string_distance(jaro_winkler, "FOO", "BAR")`. The identical strings of this example and the fully
disjoint ones of `minimal_2` are the two cases whose distance needs no approximation.
