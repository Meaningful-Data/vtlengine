:Tests: ``SC_r := DS_2#At_1`` on a viral attribute of a dataset that has identifiers.
:Manual: `Membership <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html>`__, Example 8: "The membership operator returns a Data Set having the same Identifier Components of ``ds`` and a single Measure." As in Manual example 6, ``ds_2.json`` names its dataset ``DS_1``.
:Expected: The dataset of Manual example 6, assigned to ``SC_r``.
:Erratum: The manual expects the scalar ``"P"``, but the scalar rule is for a dataset without identifiers, and ``DS_2`` has ``Id_1``, ``Id_2`` and three data points, so the general rule gives a dataset. The scalar case has its own example, *Dataset without identifiers*.
:Adaptation: ``string_var`` is written ``str_var``, as in Manual example 3.
:Team decision: `#1257 <https://github.com/Meaningful-Data/vtlengine/issues/1257>`__: as the comments on `sdmx-twg/vtl#731 <https://github.com/sdmx-twg/vtl/issues/731>`__ propose, a viral Attribute needs a propagation rule only where its values are combined, until the VTL Task Force discusses it.
