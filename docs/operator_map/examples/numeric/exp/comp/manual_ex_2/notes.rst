:Tests: ``exp`` on a component inside ``calc``, where the result replaces the measure ``Me_1``.
:Manual: `Exponential <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Exponential/index.html>`__, Example 2, output verbatim: "The operator **exp** returns *e* (base of the natural logarithm) raised to the *op*-th power."
:Expected: ``Me_1 = exp(Me_1)`` is 148.4131591025766, 2980.9579870417283 and 7.38905609893065; ``Me_2`` is unchanged.
