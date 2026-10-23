# Pivoting ex_1: why At_1 is not in the result

`DS_1` has the plain (non-viral) Attribute `At_1`, and the manual's result has only `Id_1`, `A`,
`B` and `C`. The manual's procedure, step by step:

1. VDS is a copy of `DS_1` (`Id_1`, `Id_2`, `Me_1`, `At_1`);
2. it "drops the Identifier Component *identifier* and all the Measure Components": `Id_2` and
   `Me_1` go, `Id_1` and `At_1` are left;
3. it "groups VDS by the values of the remaining Identifiers": one data point per `Id_1`. Each
   group holds several `At_1` values (`E`, `F`, `F` for `Id_1 = 1`; `E`, `E`, `F` for
   `Id_1 = 2`), and the expression gives no explicit indication about keeping `At_1`. So the
   default Attribute propagation rule of the User Manual applies: "the non-viral Attributes are
   not kept in the result and their values are not considered" ("If the virality is not
   defined, the Attribute is considered as non-viral");
4. it adds the Measures `A`, `B` and `C`, one per distinct value of `Id_2`, initialised to null;
5. each data point of `DS_1` writes its `Me_1` into the Measure named by its `Id_2`: `(1, 5, 2, 7)`
   and `(2, 3, 4, 9)`.

So the manual's expected output is consistent with its procedure, and is kept verbatim.
