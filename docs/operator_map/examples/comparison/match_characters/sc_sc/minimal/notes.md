# `match_characters`: sc_sc

Manual: *Comparison operators / Match characters*. **match_characters** returns **true** if *op* matches the regular expression, **false** otherwise (POSIX Extended Regular Expression; the pattern may be a component or a string). The strings of this example either match the whole pattern or contain no match at all, so the result does not depend on whether an implementation matches the whole string or searches inside it. `AX123` is two letters followed by three digits, so the result is `true`.

```
SC_r := match_characters("AX123", "[A-Za-z]{2}[0-9]{3}");
```
