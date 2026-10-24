:Tests: ``replace(DS_1, "ello", "i")``: ``ello`` replaced by ``i`` in every value.
:Manual: `String pattern replacement <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20pattern%20replacement/index.html>`__, Example 1, output verbatim: "Replaces all the occurrences of a specified string-pattern (``pattern1``) with another one (``pattern2``)."
:Expected: ``hello world``, ``say hello`` and ``hello!`` give ``hi world``, ``say hi`` and ``hi!``; ``he`` has no ``ello`` and is kept.
