# calc, role omitted, existing Attribute

`At_1` exists in `DS_1` as an Attribute and no role is written, so it keeps the role Attribute
(the manual: "if the component exists in the operand Data Set then it maintains its role"). Its
values are overwritten ("If the calculated Component is a Measure or an Attribute that already
exists in the input Data Set, the calculated values overwrite the original values"): `"X"` on
every data point. `Id_1` and `Me_1` are unchanged.

Triage (2026-10-23): both engines return `At_1` as a Measure. Re-derived from the rule quoted
above, the expected output stands: an existing component keeps its role when `calc` omits it.
No erratum and no team decision applies.
