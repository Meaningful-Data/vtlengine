:Tests: ``Me_1 in { 0, 3, 6, 12 }`` inside ``calc``, where the new measure ``Me_2`` takes the result.
:Manual: `Element of <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Element%20of/index.html>`__, Example 2, output verbatim: "The **in** operator returns **true** if *op* belongs to the collection, **false** otherwise."
:Expected: ``Me_2`` is true for BS and MO, and false for GZ, SQ, FJ and CQ; ``Me_1`` is kept.
