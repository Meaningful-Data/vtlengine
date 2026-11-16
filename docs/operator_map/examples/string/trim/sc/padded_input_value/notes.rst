:Tests: ``trim(SC_1)`` with the input scalar ``SC_1`` = ``"  Hello  "``, two spaces at each end, quoted in the CSV so that they are kept.
:Manual: `Whitespace removal <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Whitespace%20removal/index.html>`__: "Removes trailing or/and leading whitespace from a string. For example, ``trim("Hello ")`` gives ``"Hello"``."
:Expected: ``SC_r`` is ``Hello``, a String: both ends are trimmed.
