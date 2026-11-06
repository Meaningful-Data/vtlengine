:Tests: ``DS_1 in { 0, 3, 6, 12 }``: every measure value of ``DS_1`` against a set literal.
:Manual: `Element of <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Element%20of/index.html>`__, Example 1, output verbatim: "The **in** operator returns **true** if *op* belongs to the collection, **false** otherwise."
:Expected: ``bool_var`` is true for BS (0) and MO (6), and false for GZ, SQ, FJ and CQ (4, 9, 7 and 2).
