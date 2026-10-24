:Tests: ``DS_r := DS_1`` on a Number measure: a decimal number, ``1.5``, a negative one, ``-2.25``, and a null, which vtlengine loads, keeps and writes as they are.
:Manual: `Non-persistent assignment <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Non-persistent%20assignment/index.html>`__: "The result re is a non-persistent Data Set that has the same data structure as the Operand."
:Expected: ``DS_1`` as it is: every value comes back as it went in, the null included, and the structure keeps ``Id_1`` not nullable and ``Me_1`` nullable.
