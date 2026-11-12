:Tests: ``substr(Me_2, 1, 5)`` inside ``calc``, where the result replaces the measure ``Me_2``.
:Manual: `Sub-string extraction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Sub-string%20extraction/index.html>`__, Example 3, output verbatim: "The substring starts from the startth character of the input string and has a number of characters equal to the ``length`` parameter."
:Expected: ``Me_2`` is ``mediu``, ``short`` and ``"this "`` (with its trailing space); ``Me_1`` is unchanged.
