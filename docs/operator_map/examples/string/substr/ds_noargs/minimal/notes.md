With neither `start` nor `length`, "the substring starts from the 1st position" and ends at the end
of the input string (Sub-string extraction), so every value comes out as it went in.

- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
