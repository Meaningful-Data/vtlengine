Authored from the page's scalar semantics: "case when x1 > x2 then 2 when x1 = x2 then 0 else 5;
for x1 = 3, x2 = 0 it returns 2" (the literals 3 and 0 stand for x1 and x2). The first condition
is true, so the result is its thenOperand, 2.
