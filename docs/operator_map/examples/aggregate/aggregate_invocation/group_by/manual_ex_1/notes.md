# Aggregate invocation, ex_1: `avg ( DS_1 group by Id_1 )`

Erratum (script). The manual's `ex_1.vtl` ([Aggregate invocation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html)) is:

```
DS_r := avg ( DS_1 group by Id_1 );

# Or equivalently:
# DS_r := avg ( DS_1 group except Id_2, Id_3 )
# DS_r := avg ( DS_1#Me_1 group by Id_1 )
```

`#` does not start a comment in VTL: it is the membership operator. The manual's own
conventions ("Conventions for writing VTL Transformations") define the comments as
`/* ... */` and `// ...`, so the three lines are commented with `//` here. The transformation and
the expected output are the manual's.

`At_1` is a plain (non-viral) Attribute of `DS_1`: the default propagation rule drops it, as the
manual's result shows. Check: 2010 → (20 + 1 + 1 + 23) / 4 = 11.25, 2011 → (20 + 1 - 1 + 20) / 4
= 10, 2012 → (40 + 30) / 2 = 35.
