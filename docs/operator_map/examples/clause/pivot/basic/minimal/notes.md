# pivot with a missing combination

The manual's procedure, step by step:

1. VDS is a copy of `DS_1`;
2. `Id_2` and every Measure are dropped from VDS: only `Id_1` is left;
3. VDS is grouped by `Id_1`: the data points `1` and `2`;
4. one Measure per distinct value of `Id_2` (`A`, `B`) is added, "initialized with the
   **null** value";
5. each data point of `DS_1` writes its `Me_1` into the Measure named by its `Id_2`.

`Id_1 = 2` has no data point with `Id_2 = B`, so its `B` keeps the null of step 4.
