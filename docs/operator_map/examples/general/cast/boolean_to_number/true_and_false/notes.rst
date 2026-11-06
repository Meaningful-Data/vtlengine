:Tests: ``cast(Me_1, number)`` on a Boolean measure, an explicit cast without mask.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **boolean** to **number**: the number ``0.0`` is returned if the ``boolean`` value is false, ``1.0`` if its true."
:Expected: ``true`` gives 1.0 and ``false`` gives 0.0.
