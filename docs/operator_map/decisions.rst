============================================================================
Team decisions, errata and adaptations
============================================================================

The examples whose expected output does not come from the manual as it is written: it follows a recorded team decision, corrects an erratum of the manual, or adapts the manual's representation to vtlengine's. Each one links to the example on the page of its operator.

📌 Team decisions
--------------------------------

Expected outputs that follow a recorded team decision. The notes of an example say what the manual says and how its output was derived.

- `📌 #394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__ · `Number of days to years and days · Manual example 1 <time/daytoyear.html#ex-comp-manual-ex-1>`__ (Time operators › ``calc daytoyear(comp)``)
- `📌 #394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__ · `Number of days to years and days · Years of 365 days <time/daytoyear.html#ex-sc-years-of-365-days>`__ (Time operators › ``daytoyear(sc)``)
- `📌 #394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__ · `Number of days to months and days · Manual example 2 <time/daytomonth.html#ex-comp-manual-ex-2>`__ (Time operators › ``calc daytomonth(comp)``)
- `📌 #394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__ · `Number of days to months and days · Months of thirty days <time/daytomonth.html#ex-sc-months-of-thirty-days>`__ (Time operators › ``daytomonth(sc)``)
- `📌 #394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__ · `Years and days to number of days · Manual example 1 <time/yeartoday.html#ex-comp-manual-ex-1>`__ (Time operators › ``calc yeartoday(comp)``)
- `📌 #394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__ · `Years and days to number of days · One year and twenty days <time/yeartoday.html#ex-sc-one-year-and-twenty-days>`__ (Time operators › ``yeartoday(sc)``)
- `📌 #394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__ · `Months and days to number of days · Iso month and day durations <time/monthtoday.html#ex-comp-iso-month-and-day-durations>`__ (Time operators › ``calc monthtoday(comp)``)
- `📌 #394 <https://github.com/Meaningful-Data/vtlengine/issues/394>`__ · `Months and days to number of days · Three months and ten days <time/monthtoday.html#ex-sc-three-months-and-ten-days>`__ (Time operators › ``monthtoday(sc)``)
- `📌 #1193 <https://github.com/Meaningful-Data/vtlengine/issues/1193>`__ · `Hierarchical roll-up · Overlapping rules that agree <hierarchy/hierarchy.html#ex-condition-overlapping-rules-that-agree>`__ (Hierarchical aggregation › ``hierarchy(DS, HR condition C)``)
- `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__ · `Counting the number of data points · Single group no identifiers <aggregate/count.html#ex-ds-no-group-single-group-no-identifiers>`__ (Aggregate and Analytic operators › ``count(DS)``)
- `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__ · `Counting the number of data points · All null data point ignored <aggregate/count.html#ex-ds-group-by-all-null-data-point-ignored>`__ (Aggregate and Analytic operators › ``count(DS group by Id)``)
- `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__ · `Counting the number of data points · Drops the listed identifier <aggregate/count.html#ex-ds-group-except-drops-the-listed-identifier>`__ (Aggregate and Analytic operators › ``count(DS group except Id)``)
- `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__ · `Counting the number of data points · Quarters to years <aggregate/count.html#ex-ds-group-all-quarters-to-years>`__ (Aggregate and Analytic operators › ``count(DS group all time_agg("A"))``)
- `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__ · `Counting the number of data points · Two measures <aggregate/count.html#ex-ds-having-two-measures>`__ (Aggregate and Analytic operators › ``count(DS group by Id having cond)``)

Manual errata (corrected expected outputs, scripts or inputs)
--------------------------------------------------------------------------------------------------------------------------

Examples where the manual gets an expected output, a script or an input wrong. The example holds the corrected version, and its notes quote the manual's. These are the ones to report upstream.

- `Membership · Manual example 8 <general/membership.html#ex-viral-attribute-manual-ex-8>`__ (General purpose operators › ``DS#At (viral)``): DS\_2 has identifiers, so DS\_2#At\_1 is a dataset (the one of ex\_6), not the scalar \'P\' that the manual gives
- `Inner Join · Manual example 5 <join/inner_join.html#ex-nary-manual-ex-5>`__ (Join operators › ``inner_join(DS, DS, DS)``): Me\_3 is in both IBSC and IBSD, so stripping the alias prefixes gives two Me\_3, an error by the Behaviour section; the script adds drop IBSC#Me\_3 (all null), which gives the printed result. ds\_5.json also declares Me\_2 an Identifier, but the result has Me\_21 and Me\_22 as Measures: it is loaded as a Measure.
- `Flow to stock · Manual example 1 <time/flow_to_stock.html#ex-time-manual-ex-1>`__ (Time operators › ``flow_to_stock(DS) on time``): the expected output declares Me\_1 as String; flow\_to\_stock keeps the Integer measure of DS\_1
- `Stock to flow · Manual example 1 <time/stock_to_flow.html#ex-time-manual-ex-1>`__ (Time operators › ``stock_to_flow(DS) on time``): the input DS\_1 declares Me\_1 as String; stock\_to\_flow needs a number measure, and the values and the result are Numbers
- `Number of days to years and days · Manual example 1 <time/daytoyear.html#ex-comp-manual-ex-1>`__ (Time operators › ``calc daytoyear(comp)``): the expected output repeats Id\_1 = G on the second data point, where the input has H
- `Number of days to months and days · Manual example 2 <time/daytomonth.html#ex-comp-manual-ex-2>`__ (Time operators › ``calc daytomonth(comp)``): the expected output repeats Id\_1 = G on the second data point, where the input has H
- `Years and days to number of days · Manual example 1 <time/yeartoday.html#ex-comp-manual-ex-1>`__ (Time operators › ``calc yeartoday(comp)``): the expected data points are those of Number days to duration ex\_1 (Me\_1 = 240, Me\_2 = P0Y240D\.\.\.), not yeartoday of DS\_1
- `Aggregate invocation · Manual example 1 <aggregate/aggregate_invocation.html#ex-group-by-manual-ex-1>`__ (Aggregate and Analytic operators › ``agg(DS group by Id)``): The script comments its alternative forms with \'#\', which is not a VTL comment (the manual\'s conventions define /\* \*/ and //): they are commented with // here.
- `Aggregate invocation · Manual example 4 <aggregate/aggregate_invocation.html#ex-aggr-multiple-manual-ex-4>`__ (Aggregate and Analytic operators › ``DS[aggr Me := agg(Me), Me := agg(Me) group by Id]``): The result is named DS\_3, not DS\_r, and keeps At\_1 as a ViralAttribute; DS\_1 declares At\_1 a plain (non-viral) Attribute, which the default propagation rule drops, so DS\_r has Id\_1, Me\_2 and Me\_3 only.
- `Analytic invocation · Manual example 2 <aggregate/analytic_invocation.html#ex-partition-except-all-manual-ex-2>`__ (Aggregate and Analytic operators › ``an(over (partition except all order by Id))``): The result declares Me\_1 and Me\_2 Number: calc leaves Me\_1 Integer as in DS\_1, and rank returns integer, so both are Integer.

Adaptations
----------------------

Examples where the manual's representation of an input or of an output name was rewritten to the equivalent representation vtlengine documents (such as the ``time`` values ``2010M1/2010M12`` of an input, or the default measure name ``string_var`` of an output, which vtlengine writes ``str_var``), or values that no engine can reproduce (the manual's illustrative random draws) were left out of the expected output and are property-checked. The notes of the example keep the original. These are not errata.

- `Membership · Manual example 3 <general/membership.html#ex-attribute-manual-ex-3>`__ (General purpose operators › ``DS#At``): the default measure name string\_var is written str\_var, the name vtlengine documents (the manual\'s default names can be personalised in the implementations)
- `Membership · Manual example 6 <general/membership.html#ex-viral-attribute-manual-ex-6>`__ (General purpose operators › ``DS#At (viral)``): the default measure name string\_var is written str\_var, the name vtlengine documents (the manual\'s default names can be personalised in the implementations)
- `Membership · Manual example 8 <general/membership.html#ex-viral-attribute-manual-ex-8>`__ (General purpose operators › ``DS#At (viral)``): the default measure name string\_var is written str\_var, the name vtlengine documents (the manual\'s default names can be personalised in the implementations)
- `Random · Manual example 1 <numeric/random.html#ex-ds-manual-ex-1>`__ (Numeric operators › ``random(DS, n)``): the manual\'s illustrative random values are not reproducible; the random column is property-checked
- `Random · Manual example 2 <numeric/random.html#ex-comp-manual-ex-2>`__ (Numeric operators › ``random(comp, n)``): the manual\'s illustrative random values are not reproducible; the random column is property-checked
- `Fill time series · Manual example 1 <time/fill_time_series.html#ex-single-time-manual-ex-1>`__ (Time operators › ``fill_time_series(DS, single) on time``): the manual\'s time values YYYYMm/YYYYMm (e.g. 2010M1/2010M12), which vtlengine cannot load, are rewritten as YYYY-MM-DD/YYYY-MM-DD (2010-01-01/2010-12-31)
- `Fill time series · Manual example 2 <time/fill_time_series.html#ex-all-time-manual-ex-2>`__ (Time operators › ``fill_time_series(DS, all) on time``): the manual\'s time values YYYYMm/YYYYMm (e.g. 2010M1/2010M12), which vtlengine cannot load, are rewritten as YYYY-MM-DD/YYYY-MM-DD (2010-01-01/2010-12-31)
- `Flow to stock · Manual example 1 <time/flow_to_stock.html#ex-time-manual-ex-1>`__ (Time operators › ``flow_to_stock(DS) on time``): the manual\'s time values YYYYMm/YYYYMm (e.g. 2010M1/2010M12), which vtlengine cannot load, are rewritten as YYYY-MM-DD/YYYY-MM-DD (2010-01-01/2010-12-31)
- `Stock to flow · Manual example 1 <time/stock_to_flow.html#ex-time-manual-ex-1>`__ (Time operators › ``stock_to_flow(DS) on time``): the manual\'s time values YYYYMm/YYYYMm (e.g. 2010M1/2010M12), which vtlengine cannot load, are rewritten as YYYY-MM-DD/YYYY-MM-DD (2010-01-01/2010-12-31)
- `Time shift · Manual example 1 <time/timeshift.html#ex-time-manual-ex-1>`__ (Time operators › ``timeshift(DS, n) on time``): the manual\'s time values YYYYMm/YYYYMm (e.g. 2010M1/2010M12), which vtlengine cannot load, are rewritten as YYYY-MM-DD/YYYY-MM-DD (2010-01-01/2010-12-31)
- `Add time unit to date · One year later <time/dateadd.html#ex-sc-date-one-year-later>`__ (Time operators › ``dateadd(date, n, p)``): the page writes the year period indicator “Y” (“for years the “Y” is used”); the example writes it A, the year duration of vtlengine and of the chapter\'s duration codes
