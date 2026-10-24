:Tests: ``cast(Me_1, boolean)`` on a Number measure, an explicit cast without mask.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **number** to **boolean**: if the ``number`` absolute value is not equal to zero, then the boolean ``true`` literal is returned, in every other case the boolean ``false`` literal is returned."
:Expected: ``0.0`` gives ``false``; ``0.5`` and ``-3.25`` give ``true``.
