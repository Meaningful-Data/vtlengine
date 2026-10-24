:Tests: ``DS_2#At_1`` on a viral attribute: a new measure with its values, and ``At_1`` kept as viral.
:Manual: `Membership <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html>`__, Example 6: "If ``comp`` is an Attribute, it follows the Attribute propagation rule too", and the result has "a single Measure". The manual's ``ds_2.json`` names its dataset ``DS_1``, a slip: the example loads it as ``DS_2``, the name that the page and the script give it.
:Expected: ``Id_1``, ``Id_2``, ``str_var`` with the values of ``At_1``, and the viral ``At_1``; the values are the manual's.
:Finding: Both engines return ``Id_1``, ``Id_2`` and the viral ``At_1`` only, with no measure.
:Adaptation: ``string_var`` is written ``str_var``, as in Manual example 3.
