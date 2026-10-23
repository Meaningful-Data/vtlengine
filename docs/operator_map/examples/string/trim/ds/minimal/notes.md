`trim` removes both the leading and the trailing whitespace of each value (Whitespace removal:
"Removes trailing or/and leading whitespace from a string"; `trim("Hello ")` gives `"Hello"`).

- `"  hello  "` becomes `"hello"` and `" a b "` becomes `"a b"`.
- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.

The input CSV quotes the values, so that their leading and trailing spaces are data. The spaces
inside a value (`" a b "` has one in the middle) are never removed.
