# `in`: ds_value_domain (manual example 3)

Manual: *Comparison operators / Element of*, example 3.

The value domain `myGeoValueDomain` is not part of the example files. The page defines it in the
text that introduces the examples: "the Value Domain named myGeoValueDomain (which has the basic
scalar type *string*) defined by {"AF", "BS", "FJ", "GA", "KH", "MO", "PK", "QA", "UG"}". It is
copied to `value_domains/myGeoValueDomain.json`, which `run` receives as a value domain.

Derivation: `DS_1#Id_2` is the dataset with the identifiers Id_1 and Id_2; `in` gives the boolean
measure `bool_var`. BS, MO and FJ are elements of the value domain (true); GZ, SQ and CQ are not
(false).
