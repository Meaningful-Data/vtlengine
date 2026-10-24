:Tests: ``SC_r := DS_2#At_1`` on a viral attribute of a dataset that has identifiers.
:Manual: `Membership <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html>`__, Example 8: "The membership operator returns a Data Set having the same Identifier Components of ``ds`` and a single Measure." The manual's ``ds_2.json`` names its dataset ``DS_1``, a slip: the example loads it as ``DS_2``, the name that the page and the script give it.
:Expected: The dataset of Manual example 6, assigned to ``SC_r``: ``str_var`` with the values of ``At_1``, and the viral ``At_1``.
:Finding: As in Manual example 6, both engines return ``Id_1``, ``Id_2`` and the viral ``At_1`` only, with no measure.
:Erratum: The manual expects the scalar ``"P"``, but the scalar rule is for a dataset without identifiers, and ``DS_2`` has ``Id_1``, ``Id_2`` and three data points, so the general rule gives a dataset. The scalar case has its own example, *Dataset without identifiers*.
:Adaptation: ``string_var`` is written ``str_var``, as in Manual example 3.
