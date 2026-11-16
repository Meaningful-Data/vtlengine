:Tests: ``calc attribute At_1 := "EP"``, a constant attribute.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: "calc calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified". The calc clause page has the same calculation, ``DS_1 [ calc attribute At_1:= "EP" ]`` (its Example 2).
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands, and each gets the String Attribute ``At_1`` = ``EP``.
