Manual example `ex_1` of the if-then-else page, imported as it is:
`DS_r := if ( DS_1#Id_4 = "F" ) then DS_2 else DS_3;`. A Data Set condition with Data Sets as
thenOperand and elseOperand: variant `ds`. The only change is the name of a type: the manual's
`TimePeriod` is vtlengine's `Time_Period`.

Re-derivation (if-then-else page, Behaviour at Data Set level: the true data points of the
condition are inner joined with thenOperand, the false ones with elseOperand, and the union is
the result):

- `DS_1#Id_4` turns the identifier Id_4 into a string measure, and `= "F"` turns it into the
  boolean condition Data Set over Id_1 to Id_4: true for the six data points with Id_4 = F, false
  for the six with Id_4 = M.
- True data points with DS_2: (S,F) 25.8, (F,F) null and (I,F) 20.9 are in DS_2; (B,F), (G,F) and
  (A,F) are not, so they are left out. (A,M) of DS_2 is not a true data point either.
- False data points with DS_3: (B,M) 0.12, (G,M) 22.5 and (S,M) 23.7 are in DS_3; (F,M), (I,M) and
  (A,M) are not. (A,F) of DS_3 is not a false data point.
- The union has the six data points of the manual's result.

The manual's introduction says "Given the operand Data Sets DS_cond, DS_1 and DS_2", but the script
and the inputs use DS_1, DS_2 and DS_3. The slip is in the prose only: the example itself is
consistent, so it is not an erratum of the expected output.
