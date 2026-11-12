:Tests: the placeholder ``_`` for ``numDigit``, on a scalar: ``round(SC_1, _)`` with ``SC_1`` = 12345.6.
:Manual: `Rounding <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__: "``round ( 12345.6, _ )`` gives ``12346``".
:Expected: ``SC_r = 12346``, an Integer: the placeholder stands for an omitted ``numDigit``.
