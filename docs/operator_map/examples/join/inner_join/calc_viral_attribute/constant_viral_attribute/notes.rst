:Tests: ``calc viral attribute At_1 := "EP"``, a constant viral attribute.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: on viral attributes, "if the calc clause is present instead, it may directly replace the computation algorithm used for the propagation".
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands, and each gets the String Viral Attribute ``At_1`` = ``EP``.
:Team decision: `#1257 <https://github.com/Meaningful-Data/vtlengine/issues/1257>`__: as the comments on `sdmx-twg/vtl#731 <https://github.com/sdmx-twg/vtl/issues/731>`__ propose, a viral Attribute needs a propagation rule only where its values are combined, until the VTL Task Force discusses it.
