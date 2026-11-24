:Tests: ``cast(Me_1, string)`` on an Integer measure, without a mask: the default format.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **integer**: the ``integer`` value is printed as a sequence of digits, without any leading zeroes or thousands separators, possibly preceded by a ``-``."
:Expected: ``1500`` with no thousands separator, ``-7`` preceded by ``-``, and ``0``.
