`string_distance(jaro_winkler, "FOO", "BAR")`: the manual's own example of the operator, which it
says "gives a value between `0` and `1`" (String distances). The two strings are fully disjoint.

The expected value rests on the page's "distance" wording plus the definition distance =
1 − similarity. The page lists `jaro_winkler` among "the distance method to apply" and says that it
"Returns the Jaro–Winkler distance". It gives no formula. The usual definition of the Jaro-Winkler
distance is 1 minus the Jaro-Winkler similarity, which the Wikipedia article that the page links
says in its lead (that lead now names the metric the "Jaro–Winkler similarity"): "The score is
normalized such that 0 means an exact match and 1 means there is no similarity. The original paper
actually defined the metric in terms of similarity, so the distance is defined as the inversion of
that value (distance = 1 − similarity)" (article read on 2026-10-23).

`"FOO"` and `"BAR"` have no character in common, so the Jaro similarity is 0 (the article: "If no
matching characters are found then the strings are not similar"). The Winkler prefix boost adds
nothing either, since it is proportional to the length of the common prefix, which is 0. The
Jaro-Winkler similarity is 0, so the distance is exactly 1. The result type is number.
