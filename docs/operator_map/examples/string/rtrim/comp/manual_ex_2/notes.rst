:Tests: ``rtrim`` of a component inside ``calc``, on values that end with a space; the input quotes them, so the space is data.
:Manual: `Whitespace removal <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Whitespace%20removal/index.html>`__, Example 2, output verbatim: "Removes trailing or/and leading whitespace from a string."
:Expected: the new measure ``Me_2`` is ``hello`` and ``hi``, without the trailing space; ``Me_1`` keeps ``"hello "`` and ``"hi "``.
