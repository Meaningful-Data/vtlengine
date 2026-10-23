Manual: [Time shift](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20shift/index.html), `ex_1`.

**Input adaptation.** The page writes its *time* values at month precision, `YYYY-MM/YYYY-MM` in its words, as `2010M1/2010M12`. vtlengine cannot load them (`DataLoadError 0-3-1-6`, "Use YYYY-MM-DD/YYYY-MM-DD or YYYY or YYYY-MM", on both engines), so every value `YYYYM1/YYYYM12` of the input and of the expected output is written `YYYY-01-01/YYYY-12-31`, the same interval (the whole year) in the `YYYY-MM-DD/YYYY-MM-DD` form of the manual's "Behaviour of Time operators" and of vtlengine's data types documentation. Nothing else changes.

The manual's input `ds_1.csv`:

```
Id_1,Id_2,Me_1
A,2010M1/2010M12,"hello world"
A,2011M1/2011M12,
A,2012M1/2012M12,"say hello"
A,2013M1/2013M12,"he"
B,2010M1/2010M12,"hi, hello! "
B,2011M1/2011M12,"hi"
B,2012M1/2012M12,
B,2013M1/2013M12,"hello!"
```

The manual's expected output `ex_1.csv`:

```
Id_1,Id_2,Me_1
A,2009M1/2009M12,"hello world"
A,2010M1/2010M12,
A,2011M1/2011M12,"say hello"
A,2012M1/2012M12,"he"
B,2009M1/2009M12,"hi, hello! "
B,2010M1/2010M12,"hi"
B,2011M1/2011M12,
B,2012M1/2012M12,"hello!"
```
