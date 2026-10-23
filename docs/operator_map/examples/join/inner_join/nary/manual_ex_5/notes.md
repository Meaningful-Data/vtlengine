# inner_join: nary (manual example 5)

Manual: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html),
example 5. It chains three statements; the last one joins three operands, two of them
expressions with aliases, so it is mapped to `nary`.

## Erratum

The manual's example contradicts the page in two places. The example corrects both, and keeps
the manual's printed result (`ex_5.json`, `ex_5.csv`) as it is.

1. **`Me_3` comes from two operands of the last join.** The manual's last statement is:

   ```
   DS_r := inner_join (
      DS_4,
      IBSC[rename Id_2 to Id_21, Id_3 to Id_31, Me_2 to Me_21] as IBSC,
      IBSD[rename Id_2 to Id_22, Id_3 to Id_32, Me_2 to Me_22] as IBSD
   );
   ```

   `IBSC` and `IBSD` both have the measure `Me_3` of `DS_6`, so VDS₁ holds `IBSC#Me_3` and
   `IBSD#Me_3`. The last step of the Behaviour section then fails: "Finally, all components that
   originally appeared in multiple input data sets, are renamed by stripping their previously
   determined prefix; if this step determines a structure with homonymous components, an error
   is raised." The manual's result has a single `Me_3`, whose values are those of `IBSD#Me_3`.
   The script adds `drop IBSC#Me_3` to the last statement. `IBSC#Me_3` is null on every data
   point (every data point of `DS_6` with `Id_4` = c has a null `Me_3`), so the result is the
   manual's.

2. **`ds_5.json` gives `Me_2` the role Identifier**, while the manual's result has `Me_21` and
   `Me_22` (the renamed `Me_2`) as Measures. With the role as written they would be
   Identifiers. The name `Me_2` and the result say Measure, so the input declares `Me_2` as a
   nullable Measure. The manual's line is
   `{"name": "Me_2", "role": "Identifier", "data_type": "Integer"}`.

The importer also noted that `ds_5.json` and `ds_6.json` name their dataset `DS_r` (a
copy-paste slip with no effect: the files are imported as `DS_5` and `DS_6`, the names the
script uses).

## Derivation of the result

`CI` = `DS_6[sub Id_4 = "c"]` and `DI` = `DS_6[sub Id_4 = "d"]` (identifiers `Id_1`, `Id_2`,
measure `Me_3`):

| | Id_1, Id_2, Me_3 |
|---|---|
| CI | (1,30,null), (2,30,null), (2,20,null), (3,30,null) |
| DI | (1,10,null), (2,10,null), (3,10,50), (3,20,50) |

`IBSC` and `IBSD` join `DS_5` (`Id_2`, `Id_3`; `Me_2`) with them on `Id_2`, their only common
identifier; `Id_3` and `Id_1` each come from one operand and stay identifiers.

| | Id_1, Id_2, Id_3, Me_2, Me_3 |
|---|---|
| IBSC | (1,30,S121,18273645,null), (2,30,S121,18273645,null), (3,30,S121,18273645,null), (2,20,S2,87654321,null) |
| IBSD | (1,10,S11,12345678,null), (2,10,S11,12345678,null), (3,10,S11,12345678,50), (3,20,S2,87654321,50) |

The last join renames `Id_2`, `Id_3`, `Me_2` of each to `Id_21`, `Id_31`, `Me_21` and `Id_22`,
`Id_32`, `Me_22`, so `Id_1` is the only identifier in more than one operand: the join key.
`DS_4` has `Id_1` = 1, 2, 3 with `Me_1` = 200, 300, 100. Each key gives the product of its data
points in the three operands: 1 x 1 x 1 for `Id_1` = 1, 1 x 2 x 1 for 2, 1 x 1 x 2 for 3.

| Id_1 | Id_21 | Id_22 | Me_1 | Id_31 | Me_21 | Id_32 | Me_22 | Me_3 (IBSD) |
|---|---|---|---|---|---|---|---|---|
| 1 | 30 | 10 | 200 | S121 | 18273645 | S11 | 12345678 | null |
| 2 | 30 | 10 | 300 | S121 | 18273645 | S11 | 12345678 | null |
| 2 | 20 | 10 | 300 | S2 | 87654321 | S11 | 12345678 | null |
| 3 | 30 | 10 | 100 | S121 | 18273645 | S11 | 12345678 | 50 |
| 3 | 30 | 20 | 100 | S121 | 18273645 | S2 | 87654321 | 50 |

This is the manual's `ex_5.csv`. Only `DS_r` has an expected file; `IBSC` and `IBSD` are not
compared.
