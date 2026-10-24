:Tests: ``substr`` of a scalar whose ``start`` is past the end of the string.
:Manual: `Sub-string extraction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Sub-string%20extraction/index.html>`__: "If ``start`` is greater than the length of the input string, an empty string is extracted.", and ``substr ("abcdefghijklmnopqrstuvwxyz", 30 , 10 )`` gives ``""``.
:Expected: ``SC_r`` is the empty string. The map compares an empty value and a null as the same one, so this example cannot tell the two apart.
