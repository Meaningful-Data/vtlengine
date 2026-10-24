:Tests: ``calc viral attribute At_1 := "EP"``, a constant viral attribute.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: on viral attributes, "if the calc clause is present instead, it may directly replace the computation algorithm used for the propagation".
:Expected: the left join keeps the four keys of ``DS_1``, with ``Me_2`` null at ``(1,B)``, and each gets the String Viral Attribute ``At_1`` = ``EP``; the inputs have no attributes, so there is nothing to propagate.
