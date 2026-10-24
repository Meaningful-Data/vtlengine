:Tests: ``DS_2#Me_1`` on a dataset whose ``At_1`` is viral: the viral attribute is kept.
:Manual: `Membership <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html>`__, Example 4, output verbatim: "viral Attributes of ``ds`` are maintained in the result as viral, non-viral ones are dropped". The manual's ``ds_2.json`` names its dataset ``DS_1``, a slip: the example loads it as ``DS_2``, the name that the page and the script give it.
:Expected: ``Id_1``, ``Id_2``, ``Me_1`` and the viral ``At_1``.
