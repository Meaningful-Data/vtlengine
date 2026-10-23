Manual example `ex_1` of the Nvl page, imported as it is: `DS_r := nvl ( DS_1, 0 );`. A Data Set
and a scalar: variant `ds_sc`. The only change is the name of a type: the manual's `TimePeriod` is
vtlengine's `Time_Period`.

Re-derivation (Nvl page: returns op2 when op1 is null, otherwise op1; applied to the values of the
measures of the Data Set): Me_1 is null for S and N, so it becomes 0 there; the other four values
stay as they are. The result is the manual's.
