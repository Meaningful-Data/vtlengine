Manual: [Stock to flow](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Stock%20to%20flow/index.html), `ex_1`.

**Input adaptation.** The page writes its *time* values at month precision, `YYYY-MM/YYYY-MM` in its words, as `2010M1/2010M12`. vtlengine cannot load them (`DataLoadError 0-3-1-6`, "Use YYYY-MM-DD/YYYY-MM-DD or YYYY or YYYY-MM", on both engines), so every value `YYYYM1/YYYYM12` of the input and of the expected output is written `YYYY-01-01/YYYY-12-31`, the same interval (the whole year) in the `YYYY-MM-DD/YYYY-MM-DD` form of the manual's "Behaviour of Time operators" and of vtlengine's data types documentation. Nothing else changes.

**Erratum.** The manual's input structure (`ds_1.json`) declares `Me_1` as `String`. The operand of `stock_to_flow` is `dataset { identifier < time > _ , identifier _* , measure<number> _+ }`, the values are numbers (2.0, 7.0...) and the manual's expected structure (`ex_1.json`) has `Me_1` as `Number`, as the DS_2, DS_3 and DS_4 of the page: the input `Me_1` is `Number`.

The manual's input `ds_1.csv`:

```
Id_1,Id_2,Me_1
A,2010M1/2010M12,2.0
A,2011M1/2011M12,7.0
A,2012M1/2012M12,4.0
A,2013M1/2013M12,13.0
B,2010M1/2010M12,4.0
B,2011M1/2011M12,-4.0
B,2012M1/2012M12,-4.0
B,2013M1/2013M12,2.0
```

The manual's expected output `ex_1.csv`:

```
Id_1,Id_2,Me_1
A,2010M1/2010M12,2.0
A,2011M1/2011M12,5.0
A,2012M1/2012M12,-3.0
A,2013M1/2013M12,9.0
B,2010M1/2010M12,4.0
B,2011M1/2011M12,-8.0
B,2012M1/2012M12,0.0
B,2013M1/2013M12,6.0
```
