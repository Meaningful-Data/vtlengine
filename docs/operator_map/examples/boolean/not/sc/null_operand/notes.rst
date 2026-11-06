:Tests: ``not SC_1`` with ``SC_1`` null: the operand is null.
:Manual: `Logical negation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20negation/index.html>`__: "``not false`` gives ``true``", "``not true`` gives ``false``" and "``not null`` gives ``null``"; the page's sentence "returns **true** if *op* is **false**, otherwise **true**" has a typo ("otherwise true" would make ``not true`` true) and ignores null, and the listed rows govern.
:Expected: ``SC_r`` is null, a Boolean, as in the row "``not null`` gives ``null``".
