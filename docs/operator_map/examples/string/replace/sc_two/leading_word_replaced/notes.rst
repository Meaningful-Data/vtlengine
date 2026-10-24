:Tests: ``replace`` of a scalar, where the pattern is the first word.
:Manual: `String pattern replacement <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20pattern%20replacement/index.html>`__: "Replaces all the occurrences of a specified string-pattern (``pattern1``) with another one (``pattern2``).", as in ``replace("Hello world", "Hello", "Hi")`` gives ``"Hi world"``.
:Expected: ``SC_r`` is ``Hi world``, a String.
