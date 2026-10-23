# Unpivoting ex_2: the name of the input

The manual's `ds_2.json` for this page declares the dataset `"name": "DS_1"`, while the script
(`DS_r := DS_2 [ unpivot Id_3, Me_2 ]`) and the page text ("Given the Data Sets DS_1 and DS_2")
use `DS_2`. It is a copy-paste slip in the input file; the importer took the dataset name from
the file name, so `input/DS_2.json` declares `DS_2`. Nothing else changes.

The measures of `DS_2` are `A` (Number), `B` and `C` (Integer). The manual's constraint, "All
the measures of *op* must have the same basic scalar type, including sub-types", accepts them
because Integer is a sub-type of Number, and its expected output types the new Measure `Me_2`
as Number. The manual prints `2.1     ` (with trailing spaces) for `(6, A)`; the importer
stripped them, as for every non-String value.
