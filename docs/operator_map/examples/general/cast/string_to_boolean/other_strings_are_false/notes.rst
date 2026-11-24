:Tests: ``cast(Me_1, boolean)`` on strings that are neither ``true`` nor ``false``.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: the boolean true is returned if the string "is equal to ``"true"`` not considering the case", and "The ``boolean`` value representing false is returned in any other case."
:Expected: ``yes`` and ``1`` give ``false``: the rule compares with the text ``true`` only. The strings have no blanks around them, as the loading of blanks is not what the example tests.
