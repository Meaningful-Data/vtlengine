:Tests: ``null or null``: both operands null.
:Manual: `Logical disjunction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20disjunction/index.html>`__: "``false or null`` gives ``null``", "``true or null`` gives ``true``" and "``null or null`` gives ``null``"; the page's sentence "returns **true** if at least one of the operands is **true**, otherwise **false**" conflicts with its null rows (``false or null`` is null, not false), and the listed rows govern.
:Expected: ``SC_r`` is null, a Boolean, as in the row "``null or null`` gives ``null``".
