# Time aggregation ex_1: type of the sum

Type of the `sum`: the Sum page declares the result **number** and adds "Since **integer** is a
subtype of **number**, implementations may preserve the **integer** type for the resulting
measure or component when all operand values are **integer** and the sum is computed in
integer arithmetic", so the expected `Me_1` may be Integer or Number. The manual's `ex_1.json`
writes Integer, one of the two; the data points are the manual's, verbatim.
