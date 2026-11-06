:Tests: ``replace(Me_1, "ello", "i")`` inside ``calc``, where the new measure ``Me_2`` takes the result.
:Manual: `String pattern replacement <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20pattern%20replacement/index.html>`__, Example 2, output verbatim: "Replaces all the occurrences of a specified string-pattern (``pattern1``) with another one (``pattern2``)."
:Expected: ``Me_2`` is ``hi world``, ``say hi``, ``he`` and ``hi!``; ``Me_1`` is kept.
