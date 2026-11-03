:Tests: ``calc viral attribute At_1 := "EP"``, a constant viral attribute.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: on viral attributes, "if the calc clause is present instead, it may directly replace the computation algorithm used for the propagation".
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands, and each gets the String Viral Attribute ``At_1`` = ``EP``.
