:Tests: ``calc attribute At_1 := "EP"``, a constant attribute.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: "calc calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified". The calc clause page has the same calculation, ``DS_1 [ calc attribute At_1:= "EP" ]`` (its Example 2).
:Expected: the left join keeps the four keys of ``DS_1``, with ``Me_2`` null at ``(1,B)``, and each gets the String Attribute ``At_1`` = ``EP``.
