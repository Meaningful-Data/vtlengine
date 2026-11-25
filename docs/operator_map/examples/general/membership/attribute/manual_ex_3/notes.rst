:Tests: ``DS_1#At_1`` on a non-viral attribute: its values become a new measure, and the attribute itself is dropped.
:Manual: `Membership <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html>`__, Example 3: "A default conventional name is assigned to the new Measure depending on its type: for example ``num_var`` if the Measure is ``numeric``, ``string_var`` if it is ``string``".
:Expected: ``Id_1``, ``Id_2`` and a String measure with the values of ``At_1``; the values are the manual's.
