:Tests: ``rtrim`` on values that end with a space, which "may not be visualised" on the page; the input quotes ``"hello "`` and ``"hi "``, so the space is data.
:Manual: `Whitespace removal <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Whitespace%20removal/index.html>`__, Example 1, output verbatim: "Removes trailing or/and leading whitespace from a string."
:Expected: ``Me_1`` is ``hello`` and ``hi``, without the trailing space.
