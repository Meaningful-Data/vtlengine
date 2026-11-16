:Tests: the propagation at ``DS_1 - DS_2``, whose Measure depends on the order of the operands while the propagation does not.
:Manual: the User Manual's commutative law "holds for every operator, also for non-commutative operations like difference", and in `Typical behaviours <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html>`__ "the Attribute value is calculated by applying the proper Attribute propagation algorithm on the values of the Attributes of DP_1 and DP_2".
:Expected: ``Me_1`` 9 and 18; ``At_1`` C for (C, F) by ``when "C"`` and F for (F, F) by ``else``.
