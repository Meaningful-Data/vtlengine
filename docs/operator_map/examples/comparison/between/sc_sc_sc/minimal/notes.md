# `between`: sc_sc_sc

Manual: *Comparison operators / Between*. The operator returns **true** if *op* is greater than or equal to *from* and lower than or equal to *to*: it is a shortcut for `op >= from and op <= to`, so both limits are included. 10 is greater than or equal to 5 and lower than or equal to 10, so the result is `true`; it would be `false` if the limit were excluded.

```
SC_r := between(10, 5, 10);
```
