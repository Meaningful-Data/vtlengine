:Tests: ``substr`` of a scalar whose ``length`` overcomes what is left of the string.
:Manual: `Sub-string extraction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Sub-string%20extraction/index.html>`__: "If ``length`` is omitted or overcomes the length of the input string, the substring ends at the end of the input string.", and ``substr ("abcdefghijklmnopqrstuvwxyz", 25 , 10 )`` gives ``"yz"``.
:Expected: ``SC_r`` is ``yz``, a String: only two characters are left from the 25th.
