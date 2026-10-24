:Tests: ``instr`` of a scalar with only ``pattern``.
:Manual: `String pattern location <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20pattern%20location/index.html>`__: "The operator returns the position in the input string of a specified string ( ``pattern`` ).", as in ``instr ("abcde", "c" )`` gives ``3``.
:Expected: ``SC_r`` is 3, an Integer: ``c`` is the 3rd character of ``abcde``.
