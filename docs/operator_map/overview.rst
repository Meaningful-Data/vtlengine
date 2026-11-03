================
Overview
================

.. raw:: html

   <div class="om-legend"><span class="om-chip ok">🟢 all ok</span><span class="om-chip partial">🟡 partially implemented</span><span class="om-chip refused">🛡️ refused with an error</span><span class="om-chip missing">🔴 not implemented or crash</span><span class="om-chip diverge">🔵 engines differ</span> <span class="om-key">✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash · 📌 team decision</span></div>

How to read
----------------------

- **Pages.** The map has three parts: this overview, the operators and the definitions of the VTL-DL. The operators have a page for each block (a chapter of the manual), and the blocks a page for each operator; the definitions have a page for each definition. The sidebar, the tables and the diagrams lead to them.
- **Diagrams.** The overview and the page of a block have the map or the block in the middle, and its blocks or its operators on both sides. The page of an operator has the operator on top. When its forms use two kinds of operands or more, they are in a light box for each kind: a solid border for the forms on datasets, a dashed one for components and a dotted one for scalars, as the key above the diagram shows. Otherwise its forms are on both sides of it. The forms of a definition of the VTL-DL stand for the parts of its syntax, and are in light boxes with a thin solid border, titled after them: Signature, Body, Return type. A box with a thin, sparsely dotted border, "Used with", has the operators that it is combined with. Where the viewer supports links in diagrams, a node opens its page, and a form its section; the tables and the lists are the reliable way.
- **Statuses.** The colour of a node, and the emoji of a chip or a heading, say its status: 🟢 all its examples give the expected result on both engines; 🟡 partially implemented, a mix, or a result that is wrong on both engines; 🛡️ refused with an error: both engines raise an error of vtlengine with a code where the manual defines a result, and the code is not one of those that say that something is not implemented (the list below); 🔴 not implemented, or a crash; 🔵 the engines differ. A form takes the status of its examples, an operator that of its forms, and a block that of its operators: 🔵 when any of them is 🔵, 🟢, 🛡️ or 🔴 when all of them are, and 🟡 otherwise. The tallies count the forms of an operator, and the operators of a block.
- **Marks.** An example shows what each engine gave: ✅ a match, the expected output or the expected error; ❌ a wrong result; 🛡 a guarded error, an error that vtlengine raises on purpose, with its code, such as 🛡 2-1-15-6; 💥 a crash, an exception that vtlengine does not handle (such as ``KeyError``), or a run that did not end.
- **Examples.** An example is a folder with a ``script.vtl``, its ``input/`` data and its ``expected/`` output, or the error that it expects, and its notes: what it tests, the rule of the manual and how the result follows from it. The expected outputs come from the manual, or are derived by hand from its semantics, and never from the output of an engine. The page of an operator shows all of it, with links to the files.
- **Types.** A form takes and gives types. The page of an operator lists the operators that give what it takes ("Comes from") and those that take what it gives ("Goes to"), by block. An integer is a number, so a producer of an integer feeds an operator that takes a number.
- **Comparison.** Only the outputs that the example expects are compared: the components, their roles and types, and the data points, in any order. A number written with decimals matches a value within one unit of its last digit: an expected ``0.7`` accepts 0.6001–0.7999. The tolerance is never finer than what a double holds. A number written without decimals matches the same number, within a relative 1e-9, and one written with an exponent within a relative 1e-6. ``random`` and ``current_date`` are checked by a property of their values (for ``random``, those of the columns that are empty in the expected file), and the two engines must still return the same values.
- **Regenerating.** Run ``docs/operator_map/generate.py`` with a Python that can import vtlengine, with the compiled parser in ``src/`` (see *Building the C++ parser from source* in the contributing guide of the repository). Where systemd is available, run it under a memory cap: ``systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 python docs/operator_map/generate.py``. Elsewhere the cap is optional.

Overview diagram
--------------------------------

A node for each block, with how many operators it has, and how many of them are in each status. Where the viewer supports links in diagrams, a block opens its page. The table of the blocks below is the reliable way to every block page.

.. raw:: html

   <div class="om-diagram">

.. mermaid::
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef refused fill:#d4edda,stroke:#28a745,stroke-width:3px,stroke-dasharray: 2 2,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef block fill:#2b3a4a,stroke:#2b3a4a,color:#fff,font-weight:bold
       classDef ext fill:#fff,stroke:#666,color:#000,stroke-dasharray: 5 5
       R["VTL 2.2<br/>Reference Manual"]:::block
       chapter__general["General purpose operators<br/>7 operators · 🟢4 🟡2 🔵1"]:::diverge
       chapter__join["Join operators<br/>4 operators · 🟢1 🟡3"]:::partial
       chapter__string["String operators<br/>11 operators · 🟢10 🟡1"]:::partial
       chapter__numeric["Numeric operators<br/>18 operators · 🟢12 🟡5 🔵1"]:::diverge
       chapter__comparison["Comparison operators<br/>12 operators · 🟢12"]:::ok
       chapter__boolean["Boolean operators<br/>4 operators · 🟢4"]:::ok
       chapter__time["Time operators<br/>17 operators · 🟢11 🟡4 🛡️2"]:::partial
       chapter__set["Set operators<br/>4 operators · 🟢4"]:::ok
       chapter__hierarchy["Hierarchical aggregation<br/>1 operator · 🟡1"]:::partial
       chapter__aggregate["Aggregate and Analytic operators<br/>18 operators · 🟢5 🟡12 🔵1"]:::diverge
       chapter__validation["Data validation operators<br/>3 operators · 🟢1 🟡2"]:::partial
       chapter__conditional["Conditional operators<br/>3 operators · 🟢1 🟡2"]:::partial
       chapter__clause["Clause operators<br/>9 operators · 🟢6 🟡2 🔴1"]:::partial
       chapter__definitions["Definitions (VTL-DL)<br/>4 operators · 🟡4"]:::partial
       chapter__general --- R
       chapter__join --- R
       chapter__string --- R
       chapter__numeric --- R
       chapter__comparison --- R
       chapter__boolean --- R
       chapter__time --- R
       R --- chapter__set
       R --- chapter__hierarchy
       R --- chapter__aggregate
       R --- chapter__validation
       R --- chapter__conditional
       R --- chapter__clause
       R --- chapter__definitions
       click chapter__general href "general/index.html"
       click chapter__join href "join/index.html"
       click chapter__string href "string/index.html"
       click chapter__numeric href "numeric/index.html"
       click chapter__comparison href "comparison/index.html"
       click chapter__boolean href "boolean/index.html"
       click chapter__time href "time/index.html"
       click chapter__set href "set/index.html"
       click chapter__hierarchy href "hierarchy/index.html"
       click chapter__aggregate href "aggregate/index.html"
       click chapter__validation href "validation/index.html"
       click chapter__conditional href "conditional/index.html"
       click chapter__clause href "clause/index.html"
       click chapter__definitions href "definitions/index.html"

.. raw:: html

   </div>

Blocks
------------

The counts are over the forms of the operators of each block.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Block
     - Status
     - Operators
     - Forms
     - 🟢
     - 🟡
     - 🛡️
     - 🔴
     - 🔵
   * - :doc:`General purpose operators <general/index>`
     - 🔵
     - 7
     - 62
     - 38
     - 3
     - 2
     - 15
     - 4
   * - :doc:`Join operators <join/index>`
     - 🟡
     - 4
     - 89
     - 83
     - 5
     - 1
     - 0
     - 0
   * - :doc:`String operators <string/index>`
     - 🟡
     - 11
     - 58
     - 57
     - 1
     - 0
     - 0
     - 0
   * - :doc:`Numeric operators <numeric/index>`
     - 🔵
     - 18
     - 94
     - 82
     - 7
     - 2
     - 0
     - 3
   * - :doc:`Comparison operators <comparison/index>`
     - 🟢
     - 12
     - 67
     - 67
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Boolean operators <boolean/index>`
     - 🟢
     - 4
     - 24
     - 24
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Time operators <time/index>`
     - 🟡
     - 17
     - 54
     - 44
     - 4
     - 5
     - 1
     - 0
   * - :doc:`Set operators <set/index>`
     - 🟢
     - 4
     - 6
     - 6
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Hierarchical aggregation <hierarchy/index>`
     - 🟡
     - 1
     - 38
     - 26
     - 12
     - 0
     - 0
     - 0
   * - :doc:`Aggregate and Analytic operators <aggregate/index>`
     - 🔵
     - 18
     - 110
     - 87
     - 20
     - 2
     - 0
     - 1
   * - :doc:`Data validation operators <validation/index>`
     - 🟡
     - 3
     - 53
     - 37
     - 16
     - 0
     - 0
     - 0
   * - :doc:`Conditional operators <conditional/index>`
     - 🟡
     - 3
     - 19
     - 14
     - 3
     - 2
     - 0
     - 0
   * - :doc:`Clause operators <clause/index>`
     - 🟡
     - 9
     - 24
     - 21
     - 2
     - 0
     - 1
     - 0
   * - :doc:`Definitions (VTL-DL) <definitions/index>`
     - 🟡
     - 4
     - 29
     - 21
     - 3
     - 1
     - 4
     - 0
   * - **Total**
     - 🔵
     - 115
     - 727
     - 607
     - 76
     - 15
     - 21
     - 8

Not implemented
------------------------------

The codes of the errors of vtlengine that say that it does not implement something. A form whose examples fail with one of them on both engines is 🔴, not implemented, rather than 🛡️, refused. The list is kept by hand, in ``docs/operator_map/tool/status.py``, with the messages as the error catalogue of vtlengine writes them.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Code
     - Message
   * - ``1-3-13``
     - Unsupported VTL construct: {construct}.
   * - ``1-3-2-5``
     - At op eval: Eval in component execution is not supported, it only applies to Data Sets.
   * - ``1-3-3-8``
     - Viral propagation definition {name} is defined on value domain {value_domain}, which is not supported yet; define it on a variable instead.
   * - ``0-5-1-8``
     - Value Domain '{name}' not available for cast operator or scalar type representation or rulesets.
   * - ``0-5-1-10``
     - Invalid parameter type definition {definition}.

📌 Team decisions
--------------------------------

Expected outputs that follow a recorded team decision. The notes of an example say what the manual says and how its output was derived.

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
- `Check hierarchy · Manual example 1 <validation/check_hierarchy.html#ex-rule-manual-ex-1>`__ (Data validation operators › ``check_hierarchy(DS, HR rule C)``): rule R030 evaluates to true, so its errorcode must be null; the manual prints \"XX\"

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

Attribution
----------------------

Examples marked *manual* are taken from the VTL 2.2 Reference Manual (sdmx-twg/vtl @ 1bea1153beda0dc553767e7ca8fe16a0d51c5b34), converted to the vtlengine format.

The manual is published at `sdmx-twg.github.io/vtl/v2.2 <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/index.html>`__, and its source is the repository `sdmx-twg/vtl <https://github.com/sdmx-twg/vtl>`__.
