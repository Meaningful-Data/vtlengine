Manual: [Flow to stock](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Flow%20to%20stock/index.html), `ex_1`.

**Input adaptation.** The page writes its *time* values at month precision, `YYYY-MM/YYYY-MM` in its words, as `2010M1/2010M12`. vtlengine cannot load them (`DataLoadError 0-3-1-6`, "Use YYYY-MM-DD/YYYY-MM-DD or YYYY or YYYY-MM", on both engines), so every value `YYYYM1/YYYYM12` of the input and of the expected output is written `YYYY-01-01/YYYY-12-31`, the same interval (the whole year) in the `YYYY-MM-DD/YYYY-MM-DD` form of the manual's "Behaviour of Time operators" and of vtlengine's data types documentation. Nothing else changes.

**Erratum.** The manual's expected structure (`ex_1.json`) declares `Me_1` as `String`. `flow_to_stock` returns a Data Set with "the same Identifier, Measure and Attribute Components as the operand Data Set", whose `Me_1` is `Integer` (`ds_1.json`), and the expected values (2, 7, 4, 13...) are integers: the expected `Me_1` is `Integer`.

The manual's input `ds_1.csv`:

```
Id_1,Id_2,Me_1
A,2010M1/2010M12,2
A,2011M1/2011M12,5
A,2012M1/2012M12,-3
A,2013M1/2013M12,9
B,2010M1/2010M12,4
B,2011M1/2011M12,-8
B,2012M1/2012M12,0
B,2013M1/2013M12,6
```

The manual's expected output `ex_1.csv`:

```
Id_1,Id_2,Me_1
A,2010M1/2010M12,2
A,2011M1/2011M12,7
A,2012M1/2012M12,4
A,2013M1/2013M12,13
B,2010M1/2010M12,4
B,2011M1/2011M12,-4
B,2012M1/2012M12,-4
B,2013M1/2013M12,2
```
