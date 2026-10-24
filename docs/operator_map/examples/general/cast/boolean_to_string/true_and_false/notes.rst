:Tests: ``cast(Me_1, string)`` on a Boolean measure, an implicit conversion written as an explicit cast, whose "behavior remains the same".
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **boolean** to **string**: Either one of the string values ``"true"`` and ``"false"`` is returned, depending on the input."
:Expected: ``true`` gives the string ``true`` and ``false`` gives ``false``, in lower case as the rule writes them; a string is compared exactly.
:Finding: Both engines return ``True`` and ``False``, the spelling that vtlengine documents in ``docs/data_types.rst``: "Boolean to String: ``true`` becomes ``"True"``".
