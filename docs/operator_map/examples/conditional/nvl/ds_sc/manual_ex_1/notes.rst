:Tests: ``nvl ( DS_1, 0 )`` on a data set: each null measure value becomes 0.
:Manual: `Nvl <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Nvl/index.html>`__, Example 1, output verbatim: "The operator **nvl** returns the value from *op2* when the value from *op1* is null, otherwise it returns the value from *op1*." The page's ``TimePeriod`` type is written ``Time_Period``, as vtlengine does.
:Expected: ``Me_1`` is 0 for S and N, where it is null, and unchanged for the other four.
