:Tests: ``DS_1#Id_1``: the values of an identifier become a new measure, named for its type.
:Manual: `Membership <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html>`__, Example 2, output verbatim: "If ``comp`` is an Identifier or an Attribute Component in ``ds``, then all the existing Measures of ``ds`` are dropped in the result and a new Measure is added."
:Expected: ``Id_1``, ``Id_2`` and the measure ``int_var`` with the values of ``Id_1``; the measures and the non-viral ``At_1`` are dropped.
