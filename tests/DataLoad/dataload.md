# DataLoad specifications.
csv dialect = rfc4180
https://docs.python.org/3/library/csv.html#csv.Dialect
https://tools.ietf.org/html/rfc4180

# problem allowed null identifiers

# Nulls
,, and "" are interpreted as null.
<space> and "<space>" is not allowed as null.
None is not allowed as null.
A identifier string can have one null, only identifier string and only one record.

# Doble Quotes
Data Load, with "measure" is omitted, but with """ measure """ fails for all the types except string, and duration(but duration shuld be fixed).

# Spaces
Right and left spaces for integer,number, duration* and string are allowed.
The spaces are represented in the strings,(also in duration but this is wrong).

# Booleans
BOOLEAN, Boolean and boolean are allowed.
[0-1] and """BOOLEAN""" are not allowed.

# Integers
number.decimal[1-9] is not allowed.
number.decimal[0+] is allowed.
0.2e3 and 2000e-3  are allowed.
0.211e2 and 20e-3 aren't allowed.

# Time and Time period
Wrong letters are not allowed. Example: 2010L1/2010L12,2015L03
M01 == M1
MXXX is not allowed also M13, M14,...
THis kind of time is not allowed 2010Q2/2010M12.

# Date
yyyy/mm/dd is not allowed instead of yyyy-mm-dd
non-existent dates(30 February,...) are not allowed.