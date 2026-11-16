:Tests: ``cast(Me_1, number)`` on an Integer measure, an implicit conversion written as an explicit cast, whose "behavior remains the same".
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **integer** to **number**: a ``number`` value is returned having the integer part equal to the ``integer`` value, and the decimal part equal to zero."
:Expected: ``5`` gives 5.0 and ``-3`` gives -3.0.
