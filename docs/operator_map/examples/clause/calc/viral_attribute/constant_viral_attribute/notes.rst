:Tests: ``calc viral attribute At_1 := "EP"``: the calculation of Manual example 2 of the ``attribute`` form, with the role ``viral attribute``.
:Manual: `Calculation of a Component <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Calculation%20of%20a%20Component/index.html>`__: "The keyword viral allows controlling the virality of the calculated Attributes".
:Expected: The input, plus the Viral Attribute ``At_1`` = ``EP`` on every data point. ``calc`` combines no data points, so no propagation rule is involved: the attribute is only created.
