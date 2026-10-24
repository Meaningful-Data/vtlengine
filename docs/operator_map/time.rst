================================
🟡 Time operators
================================

54 variants: 🟢 44 · 🟡 4 · 🔴 6 · 🔵 0

← Back to :doc:`index` · vtlengine 1.9.6rc2 (851f56ae)

.. mermaid::

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef type fill:#eeeeee,color:#000
       classDef ext stroke-dasharray: 5 5
       in_time_period(("time_period")):::type
       click in_time_period "examples/types/time_period/identity/minimal/"
       in_dataset(("dataset")):::type
       click in_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       out_duration(("duration")):::type
       click out_duration "examples/types/duration/identity/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       in_time(("time")):::type
       click in_time "examples/types/time/identity/minimal/"
       in_number(("number")):::type
       click in_number "examples/types/number/identity/minimal/"
       in_integer(("integer")):::type
       click in_integer "examples/types/integer/identity/minimal/"
       in_duration(("duration")):::type
       click in_duration "examples/types/duration/identity/minimal/"
       out_time(("time")):::type
       click out_time "examples/types/time/identity/minimal/"
       ext_aggregate__aggregate_invocation["↗ Aggregate invocation (aggregate.html)"]:::ext
       click ext_aggregate__aggregate_invocation "aggregate.html"
       ext_clause__aggr["↗ Aggregation (clause.html)"]:::ext
       click ext_clause__aggr "clause.html"
       out_date(("date")):::type
       click out_date "examples/types/date/identity/minimal/"
       out_integer(("integer")):::type
       click out_integer "examples/types/integer/identity/minimal/"
       time__period_indicator["🟡 Period indicator period_indicator"]:::partial
       time__period_indicator__ds["🟢 period_indicator(DS)"]:::ok
       time__period_indicator__comp["🟢 filter period_indicator(comp)"]:::ok
       time__period_indicator__sc["🟢 period_indicator(sc)"]:::ok
       time__period_indicator__in_filter_no_operand["🔴 filter period_indicator()"]:::missing
       time__period_indicator --- time__period_indicator__ds
       time__period_indicator --- time__period_indicator__comp
       time__period_indicator --- time__period_indicator__sc
       time__period_indicator --- time__period_indicator__in_filter_no_operand
       in_time_period --> time__period_indicator
       in_dataset --> time__period_indicator
       time__period_indicator --> out_duration
       time__period_indicator --> out_dataset
       click time__period_indicator "examples/time/period_indicator/ds/manual_ex_1/"
       click time__period_indicator__ds "examples/time/period_indicator/ds/manual_ex_1/"
       click time__period_indicator__comp "examples/time/period_indicator/comp/manual_ex_2/"
       click time__period_indicator__sc "examples/time/period_indicator/sc/minimal/"
       click time__period_indicator__in_filter_no_operand "examples/time/period_indicator/in_filter_no_operand/minimal/"
       time__fill_time_series["🟢 Fill time series fill_time_series"]:::ok
       time__fill_time_series__default_tp["🟢 fill_time_series(DS) on time_period"]:::ok
       time__fill_time_series__single_tp["🟢 fill_time_series(DS, single) on time_period"]:::ok
       time__fill_time_series__all_tp["🟢 fill_time_series(DS, all) on time_period"]:::ok
       time__fill_time_series__default_date["🟢 fill_time_series(DS) on date"]:::ok
       time__fill_time_series__single_date["🟢 fill_time_series(DS, single) on date"]:::ok
       time__fill_time_series__all_date["🟢 fill_time_series(DS, all) on date"]:::ok
       time__fill_time_series__single_time["🟢 fill_time_series(DS, single) on time"]:::ok
       time__fill_time_series__all_time["🟢 fill_time_series(DS, all) on time"]:::ok
       time__fill_time_series --- time__fill_time_series__default_tp
       time__fill_time_series --- time__fill_time_series__single_tp
       time__fill_time_series --- time__fill_time_series__all_tp
       time__fill_time_series --- time__fill_time_series__default_date
       time__fill_time_series --- time__fill_time_series__single_date
       time__fill_time_series --- time__fill_time_series__all_date
       time__fill_time_series --- time__fill_time_series__single_time
       time__fill_time_series --- time__fill_time_series__all_time
       in_dataset --> time__fill_time_series
       in_time --> time__fill_time_series
       time__fill_time_series --> out_dataset
       click time__fill_time_series "examples/time/fill_time_series/default_tp/minimal/"
       click time__fill_time_series__default_tp "examples/time/fill_time_series/default_tp/minimal/"
       click time__fill_time_series__single_tp "examples/time/fill_time_series/single_tp/manual_ex_5/"
       click time__fill_time_series__all_tp "examples/time/fill_time_series/all_tp/manual_ex_6/"
       click time__fill_time_series__default_date "examples/time/fill_time_series/default_date/minimal/"
       click time__fill_time_series__single_date "examples/time/fill_time_series/single_date/manual_ex_3/"
       click time__fill_time_series__all_date "examples/time/fill_time_series/all_date/manual_ex_4/"
       click time__fill_time_series__single_time "examples/time/fill_time_series/single_time/manual_ex_1/"
       click time__fill_time_series__all_time "examples/time/fill_time_series/all_time/manual_ex_2/"
       time__flow_to_stock["🟢 Flow to stock flow_to_stock"]:::ok
       time__flow_to_stock__time_period["🟢 flow_to_stock(DS) on time_period"]:::ok
       time__flow_to_stock__date["🟢 flow_to_stock(DS) on date"]:::ok
       time__flow_to_stock__time["🟢 flow_to_stock(DS) on time"]:::ok
       time__flow_to_stock --- time__flow_to_stock__time_period
       time__flow_to_stock --- time__flow_to_stock__date
       time__flow_to_stock --- time__flow_to_stock__time
       in_dataset --> time__flow_to_stock
       in_time --> time__flow_to_stock
       in_number --> time__flow_to_stock
       time__flow_to_stock --> out_dataset
       click time__flow_to_stock "examples/time/flow_to_stock/time_period/manual_ex_3/"
       click time__flow_to_stock__time_period "examples/time/flow_to_stock/time_period/manual_ex_3/"
       click time__flow_to_stock__date "examples/time/flow_to_stock/date/manual_ex_2/"
       click time__flow_to_stock__time "examples/time/flow_to_stock/time/manual_ex_1/"
       time__stock_to_flow["🟢 Stock to flow stock_to_flow"]:::ok
       time__stock_to_flow__time_period["🟢 stock_to_flow(DS) on time_period"]:::ok
       time__stock_to_flow__date["🟢 stock_to_flow(DS) on date"]:::ok
       time__stock_to_flow__time["🟢 stock_to_flow(DS) on time"]:::ok
       time__stock_to_flow --- time__stock_to_flow__time_period
       time__stock_to_flow --- time__stock_to_flow__date
       time__stock_to_flow --- time__stock_to_flow__time
       in_dataset --> time__stock_to_flow
       in_time --> time__stock_to_flow
       in_number --> time__stock_to_flow
       time__stock_to_flow --> out_dataset
       click time__stock_to_flow "examples/time/stock_to_flow/time_period/manual_ex_3/"
       click time__stock_to_flow__time_period "examples/time/stock_to_flow/time_period/manual_ex_3/"
       click time__stock_to_flow__date "examples/time/stock_to_flow/date/manual_ex_2/"
       click time__stock_to_flow__time "examples/time/stock_to_flow/time/manual_ex_1/"
       time__timeshift["🟢 Time shift timeshift"]:::ok
       time__timeshift__time_period["🟢 timeshift(DS, n) on time_period"]:::ok
       time__timeshift__date["🟢 timeshift(DS, n) on date"]:::ok
       time__timeshift__time["🟢 timeshift(DS, n) on time"]:::ok
       time__timeshift --- time__timeshift__time_period
       time__timeshift --- time__timeshift__date
       time__timeshift --- time__timeshift__time
       in_dataset --> time__timeshift
       in_time --> time__timeshift
       in_integer --> time__timeshift
       time__timeshift --> out_dataset
       click time__timeshift "examples/time/timeshift/time_period/manual_ex_3/"
       click time__timeshift__time_period "examples/time/timeshift/time_period/manual_ex_3/"
       click time__timeshift__date "examples/time/timeshift/date/manual_ex_2/"
       click time__timeshift__time "examples/time/timeshift/time/manual_ex_1/"
       time__time_agg["🟡 Time aggregation time_agg"]:::partial
       time__time_agg__group_all["🟢 sum(DS group all time_agg(p))"]:::ok
       time__time_agg__ds["🟢 time_agg(p, DS)"]:::ok
       time__time_agg__comp["🟢 calc time_agg(p, comp)"]:::ok
       time__time_agg__sc_time_period["🟢 time_agg(p, sc) on time_period"]:::ok
       time__time_agg__sc_date["🔴 time_agg(p, sc) on date"]:::missing
       time__time_agg__period_from["🟢 time_agg(p, from, sc)"]:::ok
       time__time_agg__first["🟢 time_agg(p, sc, first) on date"]:::ok
       time__time_agg__last["🟢 time_agg(p, sc, last) on date"]:::ok
       time__time_agg --- time__time_agg__group_all
       time__time_agg --- time__time_agg__ds
       time__time_agg --- time__time_agg__comp
       time__time_agg --- time__time_agg__sc_time_period
       time__time_agg --- time__time_agg__sc_date
       time__time_agg --- time__time_agg__period_from
       time__time_agg --- time__time_agg__first
       time__time_agg --- time__time_agg__last
       in_time --> time__time_agg
       in_duration --> time__time_agg
       in_dataset --> time__time_agg
       time__time_agg --> out_time
       time__time_agg --> out_dataset
       time__time_agg -. "used in group all" .-> ext_aggregate__aggregate_invocation
       time__time_agg -. "used in group all" .-> ext_clause__aggr
       click time__time_agg "examples/time/time_agg/group_all/manual_ex_1/"
       click time__time_agg__group_all "examples/time/time_agg/group_all/manual_ex_1/"
       click time__time_agg__ds "examples/time/time_agg/ds/minimal/"
       click time__time_agg__comp "examples/time/time_agg/comp/minimal/"
       click time__time_agg__sc_time_period "examples/time/time_agg/sc_time_period/minimal/"
       click time__time_agg__sc_date "examples/time/time_agg/sc_date/minimal/"
       click time__time_agg__period_from "examples/time/time_agg/period_from/minimal/"
       click time__time_agg__first "examples/time/time_agg/first/minimal/"
       click time__time_agg__last "examples/time/time_agg/last/minimal/"
       time__current_date["🟢 Actual time current_date"]:::ok
       time__current_date__sc["🟢 current_date()"]:::ok
       time__current_date --- time__current_date__sc
       time__current_date --> out_date
       click time__current_date "examples/time/current_date/sc/minimal/"
       click time__current_date__sc "examples/time/current_date/sc/minimal/"
       time__datediff["🟢 Days between two dates datediff"]:::ok
       time__datediff__comp_comp["🟢 calc datediff(comp, comp)"]:::ok
       time__datediff__sc_date["🟢 datediff(date, date)"]:::ok
       time__datediff__sc_time_period["🟢 datediff(time_period, time_period)"]:::ok
       time__datediff__sc_mixed["🟢 datediff(time_period, date)"]:::ok
       time__datediff --- time__datediff__comp_comp
       time__datediff --- time__datediff__sc_date
       time__datediff --- time__datediff__sc_time_period
       time__datediff --- time__datediff__sc_mixed
       in_time --> time__datediff
       time__datediff --> out_integer
       click time__datediff "examples/time/datediff/comp_comp/manual_ex_1/"
       click time__datediff__comp_comp "examples/time/datediff/comp_comp/manual_ex_1/"
       click time__datediff__sc_date "examples/time/datediff/sc_date/minimal/"
       click time__datediff__sc_time_period "examples/time/datediff/sc_time_period/minimal/"
       click time__datediff__sc_mixed "examples/time/datediff/sc_mixed/minimal/"
       time__dateadd["🟢 Add time unit to date dateadd"]:::ok
       time__dateadd__ds["🟢 dateadd(DS, n, p)"]:::ok
       time__dateadd__comp["🟢 calc dateadd(comp, n, p)"]:::ok
       time__dateadd__sc_date["🟢 dateadd(date, n, p)"]:::ok
       time__dateadd__sc_time_period["🟢 dateadd(time_period, n, p)"]:::ok
       time__dateadd --- time__dateadd__ds
       time__dateadd --- time__dateadd__comp
       time__dateadd --- time__dateadd__sc_date
       time__dateadd --- time__dateadd__sc_time_period
       in_time --> time__dateadd
       in_integer --> time__dateadd
       in_duration --> time__dateadd
       in_dataset --> time__dateadd
       time__dateadd --> out_time
       time__dateadd --> out_dataset
       click time__dateadd "examples/time/dateadd/ds/minimal/"
       click time__dateadd__ds "examples/time/dateadd/ds/minimal/"
       click time__dateadd__comp "examples/time/dateadd/comp/manual_ex_1/"
       click time__dateadd__sc_date "examples/time/dateadd/sc_date/minimal/"
       click time__dateadd__sc_time_period "examples/time/dateadd/sc_time_period/minimal/"
       time__getyear["🟢 Extract the year getyear"]:::ok
       time__getyear__comp["🟢 calc getyear(comp)"]:::ok
       time__getyear__sc["🟢 getyear(sc)"]:::ok
       time__getyear --- time__getyear__comp
       time__getyear --- time__getyear__sc
       in_time --> time__getyear
       time__getyear --> out_integer
       click time__getyear "examples/time/getyear/comp/minimal/"
       click time__getyear__comp "examples/time/getyear/comp/minimal/"
       click time__getyear__sc "examples/time/getyear/sc/minimal/"
       time__getmonth["🟢 Extract the month getmonth"]:::ok
       time__getmonth__comp["🟢 calc getmonth(comp)"]:::ok
       time__getmonth__sc["🟢 getmonth(sc)"]:::ok
       time__getmonth --- time__getmonth__comp
       time__getmonth --- time__getmonth__sc
       in_time --> time__getmonth
       time__getmonth --> out_integer
       click time__getmonth "examples/time/getmonth/comp/manual_ex_1/"
       click time__getmonth__comp "examples/time/getmonth/comp/manual_ex_1/"
       click time__getmonth__sc "examples/time/getmonth/sc/minimal/"
       time__dayofmonth["🟢 Extract the day of the month dayofmonth"]:::ok
       time__dayofmonth__comp["🟢 calc dayofmonth(comp)"]:::ok
       time__dayofmonth__sc["🟢 dayofmonth(sc)"]:::ok
       time__dayofmonth --- time__dayofmonth__comp
       time__dayofmonth --- time__dayofmonth__sc
       in_time --> time__dayofmonth
       time__dayofmonth --> out_integer
       click time__dayofmonth "examples/time/dayofmonth/comp/minimal/"
       click time__dayofmonth__comp "examples/time/dayofmonth/comp/minimal/"
       click time__dayofmonth__sc "examples/time/dayofmonth/sc/minimal/"
       time__dayofyear["🟢 Extract the day of the year dayofyear"]:::ok
       time__dayofyear__comp["🟢 calc dayofyear(comp)"]:::ok
       time__dayofyear__sc["🟢 dayofyear(sc)"]:::ok
       time__dayofyear --- time__dayofyear__comp
       time__dayofyear --- time__dayofyear__sc
       in_time --> time__dayofyear
       time__dayofyear --> out_integer
       click time__dayofyear "examples/time/dayofyear/comp/minimal/"
       click time__dayofyear__comp "examples/time/dayofyear/comp/minimal/"
       click time__dayofyear__sc "examples/time/dayofyear/sc/minimal/"
       time__daytoyear["🟡 Number of days to years and days daytoyear"]:::partial
       time__daytoyear__comp["🟡 calc daytoyear(comp)"]:::partial
       time__daytoyear__sc["🟡 daytoyear(sc)"]:::partial
       time__daytoyear --- time__daytoyear__comp
       time__daytoyear --- time__daytoyear__sc
       in_integer --> time__daytoyear
       time__daytoyear --> out_duration
       click time__daytoyear "examples/time/daytoyear/comp/manual_ex_1/"
       click time__daytoyear__comp "examples/time/daytoyear/comp/manual_ex_1/"
       click time__daytoyear__sc "examples/time/daytoyear/sc/minimal/"
       time__daytomonth["🟡 Number of days to months and days daytomonth"]:::partial
       time__daytomonth__comp["🟡 calc daytomonth(comp)"]:::partial
       time__daytomonth__sc["🟡 daytomonth(sc)"]:::partial
       time__daytomonth --- time__daytomonth__comp
       time__daytomonth --- time__daytomonth__sc
       in_integer --> time__daytomonth
       time__daytomonth --> out_duration
       click time__daytomonth "examples/time/daytomonth/comp/manual_ex_2/"
       click time__daytomonth__comp "examples/time/daytomonth/comp/manual_ex_2/"
       click time__daytomonth__sc "examples/time/daytomonth/sc/minimal/"
       time__yeartoday["🔴 Years and days to number of days yeartoday"]:::missing
       time__yeartoday__comp["🔴 calc yeartoday(comp)"]:::missing
       time__yeartoday__sc["🔴 yeartoday(sc)"]:::missing
       time__yeartoday --- time__yeartoday__comp
       time__yeartoday --- time__yeartoday__sc
       in_duration --> time__yeartoday
       time__yeartoday --> out_integer
       click time__yeartoday "examples/time/yeartoday/comp/manual_ex_1/"
       click time__yeartoday__comp "examples/time/yeartoday/comp/manual_ex_1/"
       click time__yeartoday__sc "examples/time/yeartoday/sc/minimal/"
       time__monthtoday["🔴 Months and days to number of days monthtoday"]:::missing
       time__monthtoday__comp["🔴 calc monthtoday(comp)"]:::missing
       time__monthtoday__sc["🔴 monthtoday(sc)"]:::missing
       time__monthtoday --- time__monthtoday__comp
       time__monthtoday --- time__monthtoday__sc
       in_duration --> time__monthtoday
       time__monthtoday --> out_integer
       click time__monthtoday "examples/time/monthtoday/comp/minimal/"
       click time__monthtoday__comp "examples/time/monthtoday/comp/minimal/"
       click time__monthtoday__sc "examples/time/monthtoday/sc/minimal/"

🟡 Period indicator ``period_indicator``
------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Period%20indicator/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``period_indicator(DS)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/period_indicator/ds/manual_ex_1/script.vtl>`__ · `input <examples/time/period_indicator/ds/manual_ex_1/input/>`__ · `expected <examples/time/period_indicator/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``filter period_indicator(comp)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/period_indicator/comp/manual_ex_2/script.vtl>`__ · `input <examples/time/period_indicator/comp/manual_ex_2/input/>`__ · `expected <examples/time/period_indicator/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``period_indicator(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/period_indicator/sc/minimal/script.vtl>`__ · `expected <examples/time/period_indicator/sc/minimal/expected/>`__ · `notes <examples/time/period_indicator/sc/minimal/notes.md>`__
   * - | ``filter period_indicator()``
       | minimal
     - 🔴
     - 💥
     - 💥
     - IndexError: list index out of range
     - `script <examples/time/period_indicator/in_filter_no_operand/minimal/script.vtl>`__ · `input <examples/time/period_indicator/in_filter_no_operand/minimal/input/>`__ · `expected <examples/time/period_indicator/in_filter_no_operand/minimal/expected/>`__ · `notes <examples/time/period_indicator/in_filter_no_operand/minimal/notes.md>`__

🟢 Fill time series ``fill_time_series``
------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Fill%20time%20series/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``fill_time_series(DS) on time_period``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/default_tp/minimal/script.vtl>`__ · `input <examples/time/fill_time_series/default_tp/minimal/input/>`__ · `expected <examples/time/fill_time_series/default_tp/minimal/expected/>`__ · `notes <examples/time/fill_time_series/default_tp/minimal/notes.md>`__
   * - | ``fill_time_series(DS, single) on time_period``
       | manual\_ex\_5
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/single_tp/manual_ex_5/script.vtl>`__ · `input <examples/time/fill_time_series/single_tp/manual_ex_5/input/>`__ · `expected <examples/time/fill_time_series/single_tp/manual_ex_5/expected/>`__ · *manual*
   * - | ``fill_time_series(DS, single) on time_period``
       | manual\_ex\_7
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/single_tp/manual_ex_7/script.vtl>`__ · `input <examples/time/fill_time_series/single_tp/manual_ex_7/input/>`__ · `expected <examples/time/fill_time_series/single_tp/manual_ex_7/expected/>`__ · *manual*
   * - | ``fill_time_series(DS, all) on time_period``
       | manual\_ex\_6
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/all_tp/manual_ex_6/script.vtl>`__ · `input <examples/time/fill_time_series/all_tp/manual_ex_6/input/>`__ · `expected <examples/time/fill_time_series/all_tp/manual_ex_6/expected/>`__ · *manual*
   * - | ``fill_time_series(DS, all) on time_period``
       | manual\_ex\_8
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/all_tp/manual_ex_8/script.vtl>`__ · `input <examples/time/fill_time_series/all_tp/manual_ex_8/input/>`__ · `expected <examples/time/fill_time_series/all_tp/manual_ex_8/expected/>`__ · *manual*
   * - | ``fill_time_series(DS) on date``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/default_date/minimal/script.vtl>`__ · `input <examples/time/fill_time_series/default_date/minimal/input/>`__ · `expected <examples/time/fill_time_series/default_date/minimal/expected/>`__ · `notes <examples/time/fill_time_series/default_date/minimal/notes.md>`__
   * - | ``fill_time_series(DS, single) on date``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/single_date/manual_ex_3/script.vtl>`__ · `input <examples/time/fill_time_series/single_date/manual_ex_3/input/>`__ · `expected <examples/time/fill_time_series/single_date/manual_ex_3/expected/>`__ · *manual*
   * - | ``fill_time_series(DS, all) on date``
       | manual\_ex\_4
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/all_date/manual_ex_4/script.vtl>`__ · `input <examples/time/fill_time_series/all_date/manual_ex_4/input/>`__ · `expected <examples/time/fill_time_series/all_date/manual_ex_4/expected/>`__ · *manual*
   * - | ``fill_time_series(DS, single) on time``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/single_time/manual_ex_1/script.vtl>`__ · `input <examples/time/fill_time_series/single_time/manual_ex_1/input/>`__ · `expected <examples/time/fill_time_series/single_time/manual_ex_1/expected/>`__ · `notes <examples/time/fill_time_series/single_time/manual_ex_1/notes.md>`__ · *manual* · adapted
   * - | ``fill_time_series(DS, all) on time``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/fill_time_series/all_time/manual_ex_2/script.vtl>`__ · `input <examples/time/fill_time_series/all_time/manual_ex_2/input/>`__ · `expected <examples/time/fill_time_series/all_time/manual_ex_2/expected/>`__ · `notes <examples/time/fill_time_series/all_time/manual_ex_2/notes.md>`__ · *manual* · adapted

🟢 Flow to stock ``flow_to_stock``
------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Flow%20to%20stock/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``flow_to_stock(DS) on time_period``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/flow_to_stock/time_period/manual_ex_3/script.vtl>`__ · `input <examples/time/flow_to_stock/time_period/manual_ex_3/input/>`__ · `expected <examples/time/flow_to_stock/time_period/manual_ex_3/expected/>`__ · *manual*
   * - | ``flow_to_stock(DS) on time_period``
       | manual\_ex\_4
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/flow_to_stock/time_period/manual_ex_4/script.vtl>`__ · `input <examples/time/flow_to_stock/time_period/manual_ex_4/input/>`__ · `expected <examples/time/flow_to_stock/time_period/manual_ex_4/expected/>`__ · *manual*
   * - | ``flow_to_stock(DS) on date``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/flow_to_stock/date/manual_ex_2/script.vtl>`__ · `input <examples/time/flow_to_stock/date/manual_ex_2/input/>`__ · `expected <examples/time/flow_to_stock/date/manual_ex_2/expected/>`__ · *manual*
   * - | ``flow_to_stock(DS) on time``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/flow_to_stock/time/manual_ex_1/script.vtl>`__ · `input <examples/time/flow_to_stock/time/manual_ex_1/input/>`__ · `expected <examples/time/flow_to_stock/time/manual_ex_1/expected/>`__ · `notes <examples/time/flow_to_stock/time/manual_ex_1/notes.md>`__ · *manual* · erratum · adapted

🟢 Stock to flow ``stock_to_flow``
------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Stock%20to%20flow/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``stock_to_flow(DS) on time_period``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/stock_to_flow/time_period/manual_ex_3/script.vtl>`__ · `input <examples/time/stock_to_flow/time_period/manual_ex_3/input/>`__ · `expected <examples/time/stock_to_flow/time_period/manual_ex_3/expected/>`__ · *manual*
   * - | ``stock_to_flow(DS) on time_period``
       | manual\_ex\_4
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/stock_to_flow/time_period/manual_ex_4/script.vtl>`__ · `input <examples/time/stock_to_flow/time_period/manual_ex_4/input/>`__ · `expected <examples/time/stock_to_flow/time_period/manual_ex_4/expected/>`__ · *manual*
   * - | ``stock_to_flow(DS) on date``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/stock_to_flow/date/manual_ex_2/script.vtl>`__ · `input <examples/time/stock_to_flow/date/manual_ex_2/input/>`__ · `expected <examples/time/stock_to_flow/date/manual_ex_2/expected/>`__ · *manual*
   * - | ``stock_to_flow(DS) on time``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/stock_to_flow/time/manual_ex_1/script.vtl>`__ · `input <examples/time/stock_to_flow/time/manual_ex_1/input/>`__ · `expected <examples/time/stock_to_flow/time/manual_ex_1/expected/>`__ · `notes <examples/time/stock_to_flow/time/manual_ex_1/notes.md>`__ · *manual* · erratum · adapted

🟢 Time shift ``timeshift``
----------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20shift/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``timeshift(DS, n) on time_period``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/timeshift/time_period/manual_ex_3/script.vtl>`__ · `input <examples/time/timeshift/time_period/manual_ex_3/input/>`__ · `expected <examples/time/timeshift/time_period/manual_ex_3/expected/>`__ · *manual*
   * - | ``timeshift(DS, n) on time_period``
       | manual\_ex\_4
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/timeshift/time_period/manual_ex_4/script.vtl>`__ · `input <examples/time/timeshift/time_period/manual_ex_4/input/>`__ · `expected <examples/time/timeshift/time_period/manual_ex_4/expected/>`__ · *manual*
   * - | ``timeshift(DS, n) on date``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/timeshift/date/manual_ex_2/script.vtl>`__ · `input <examples/time/timeshift/date/manual_ex_2/input/>`__ · `expected <examples/time/timeshift/date/manual_ex_2/expected/>`__ · *manual*
   * - | ``timeshift(DS, n) on time``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/timeshift/time/manual_ex_1/script.vtl>`__ · `input <examples/time/timeshift/time/manual_ex_1/input/>`__ · `expected <examples/time/timeshift/time/manual_ex_1/expected/>`__ · `notes <examples/time/timeshift/time/manual_ex_1/notes.md>`__ · *manual* · adapted

🟡 Time aggregation ``time_agg``
--------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``sum(DS group all time_agg(p))``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/time_agg/group_all/manual_ex_1/script.vtl>`__ · `input <examples/time/time_agg/group_all/manual_ex_1/input/>`__ · `expected <examples/time/time_agg/group_all/manual_ex_1/expected/>`__ · `notes <examples/time/time_agg/group_all/manual_ex_1/notes.md>`__ · *manual*
   * - | ``time_agg(p, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/time_agg/ds/minimal/script.vtl>`__ · `input <examples/time/time_agg/ds/minimal/input/>`__ · `expected <examples/time/time_agg/ds/minimal/expected/>`__ · `notes <examples/time/time_agg/ds/minimal/notes.md>`__
   * - | ``calc time_agg(p, comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/time_agg/comp/minimal/script.vtl>`__ · `input <examples/time/time_agg/comp/minimal/input/>`__ · `expected <examples/time/time_agg/comp/minimal/expected/>`__ · `notes <examples/time/time_agg/comp/minimal/notes.md>`__
   * - | ``time_agg(p, sc) on time_period``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/time_agg/sc_time_period/minimal/script.vtl>`__ · `expected <examples/time/time_agg/sc_time_period/minimal/expected/>`__ · `notes <examples/time/time_agg/sc_time_period/minimal/notes.md>`__
   * - | ``time_agg(p, sc) on date``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-19-11: At op time\_agg: If used over a Date, first/last parameter must be declared. Please check transformation with output Dataset DS\_r
     - `script <examples/time/time_agg/sc_date/minimal/script.vtl>`__ · `expected <examples/time/time_agg/sc_date/minimal/expected/>`__ · `notes <examples/time/time_agg/sc_date/minimal/notes.md>`__
   * - | ``time_agg(p, from, sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/time_agg/period_from/minimal/script.vtl>`__ · `expected <examples/time/time_agg/period_from/minimal/expected/>`__ · `notes <examples/time/time_agg/period_from/minimal/notes.md>`__
   * - | ``time_agg(p, sc, first) on date``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/time_agg/first/minimal/script.vtl>`__ · `expected <examples/time/time_agg/first/minimal/expected/>`__ · `notes <examples/time/time_agg/first/minimal/notes.md>`__
   * - | ``time_agg(p, sc, last) on date``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/time_agg/last/minimal/script.vtl>`__ · `expected <examples/time/time_agg/last/minimal/expected/>`__ · `notes <examples/time/time_agg/last/minimal/notes.md>`__

🟢 Actual time ``current_date``
------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Actual%20time/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``current_date()``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/current_date/sc/minimal/script.vtl>`__ · `expected <examples/time/current_date/sc/minimal/expected/>`__ · `notes <examples/time/current_date/sc/minimal/notes.md>`__

🟢 Days between two dates ``datediff``
--------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Days%20between%20two%20dates/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc datediff(comp, comp)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/datediff/comp_comp/manual_ex_1/script.vtl>`__ · `input <examples/time/datediff/comp_comp/manual_ex_1/input/>`__ · `expected <examples/time/datediff/comp_comp/manual_ex_1/expected/>`__ · *manual*
   * - | ``datediff(date, date)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/datediff/sc_date/minimal/script.vtl>`__ · `expected <examples/time/datediff/sc_date/minimal/expected/>`__ · `notes <examples/time/datediff/sc_date/minimal/notes.md>`__
   * - | ``datediff(time_period, time_period)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/datediff/sc_time_period/minimal/script.vtl>`__ · `expected <examples/time/datediff/sc_time_period/minimal/expected/>`__ · `notes <examples/time/datediff/sc_time_period/minimal/notes.md>`__
   * - | ``datediff(time_period, date)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/datediff/sc_mixed/minimal/script.vtl>`__ · `expected <examples/time/datediff/sc_mixed/minimal/expected/>`__ · `notes <examples/time/datediff/sc_mixed/minimal/notes.md>`__

🟢 Add time unit to date ``dateadd``
----------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Add%20time%20unit%20to%20date/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``dateadd(DS, n, p)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dateadd/ds/minimal/script.vtl>`__ · `input <examples/time/dateadd/ds/minimal/input/>`__ · `expected <examples/time/dateadd/ds/minimal/expected/>`__ · `notes <examples/time/dateadd/ds/minimal/notes.md>`__
   * - | ``calc dateadd(comp, n, p)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dateadd/comp/manual_ex_1/script.vtl>`__ · `input <examples/time/dateadd/comp/manual_ex_1/input/>`__ · `expected <examples/time/dateadd/comp/manual_ex_1/expected/>`__ · *manual*
   * - | ``dateadd(date, n, p)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dateadd/sc_date/minimal/script.vtl>`__ · `expected <examples/time/dateadd/sc_date/minimal/expected/>`__ · `notes <examples/time/dateadd/sc_date/minimal/notes.md>`__
   * - | ``dateadd(date, n, p)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dateadd/sc_date/minimal_2/script.vtl>`__ · `expected <examples/time/dateadd/sc_date/minimal_2/expected/>`__ · `notes <examples/time/dateadd/sc_date/minimal_2/notes.md>`__ · adapted
   * - | ``dateadd(time_period, n, p)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dateadd/sc_time_period/minimal/script.vtl>`__ · `expected <examples/time/dateadd/sc_time_period/minimal/expected/>`__ · `notes <examples/time/dateadd/sc_time_period/minimal/notes.md>`__

🟢 Extract the year ``getyear``
------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Extract%20time%20period/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc getyear(comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/getyear/comp/minimal/script.vtl>`__ · `input <examples/time/getyear/comp/minimal/input/>`__ · `expected <examples/time/getyear/comp/minimal/expected/>`__ · `notes <examples/time/getyear/comp/minimal/notes.md>`__
   * - | ``getyear(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/getyear/sc/minimal/script.vtl>`__ · `expected <examples/time/getyear/sc/minimal/expected/>`__ · `notes <examples/time/getyear/sc/minimal/notes.md>`__

🟢 Extract the month ``getmonth``
----------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Extract%20time%20period/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc getmonth(comp)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/getmonth/comp/manual_ex_1/script.vtl>`__ · `input <examples/time/getmonth/comp/manual_ex_1/input/>`__ · `expected <examples/time/getmonth/comp/manual_ex_1/expected/>`__ · *manual*
   * - | ``getmonth(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/getmonth/sc/minimal/script.vtl>`__ · `expected <examples/time/getmonth/sc/minimal/expected/>`__ · `notes <examples/time/getmonth/sc/minimal/notes.md>`__

🟢 Extract the day of the month ``dayofmonth``
------------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Extract%20time%20period/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc dayofmonth(comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dayofmonth/comp/minimal/script.vtl>`__ · `input <examples/time/dayofmonth/comp/minimal/input/>`__ · `expected <examples/time/dayofmonth/comp/minimal/expected/>`__ · `notes <examples/time/dayofmonth/comp/minimal/notes.md>`__
   * - | ``dayofmonth(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dayofmonth/sc/minimal/script.vtl>`__ · `expected <examples/time/dayofmonth/sc/minimal/expected/>`__ · `notes <examples/time/dayofmonth/sc/minimal/notes.md>`__

🟢 Extract the day of the year ``dayofyear``
--------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Extract%20time%20period/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc dayofyear(comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dayofyear/comp/minimal/script.vtl>`__ · `input <examples/time/dayofyear/comp/minimal/input/>`__ · `expected <examples/time/dayofyear/comp/minimal/expected/>`__ · `notes <examples/time/dayofyear/comp/minimal/notes.md>`__
   * - | ``dayofyear(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/time/dayofyear/sc/minimal/script.vtl>`__ · `expected <examples/time/dayofyear/sc/minimal/expected/>`__ · `notes <examples/time/dayofyear/sc/minimal/notes.md>`__

🟡 Number of days to years and days ``daytoyear``
------------------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Number%20days%20to%20duration/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc daytoyear(comp)``
       | manual\_ex\_1
     - 🟡
     - ❌
     - ❌
     - type of Me\_2 expected Duration got String
     - `script <examples/time/daytoyear/comp/manual_ex_1/script.vtl>`__ · `input <examples/time/daytoyear/comp/manual_ex_1/input/>`__ · `expected <examples/time/daytoyear/comp/manual_ex_1/expected/>`__ · `notes <examples/time/daytoyear/comp/manual_ex_1/notes.md>`__ · *manual* · erratum
   * - | ``daytoyear(sc)``
       | minimal
     - 🟡
     - ❌
     - ❌
     - type expected Duration got String
     - `script <examples/time/daytoyear/sc/minimal/script.vtl>`__ · `expected <examples/time/daytoyear/sc/minimal/expected/>`__ · `notes <examples/time/daytoyear/sc/minimal/notes.md>`__

🟡 Number of days to months and days ``daytomonth``
----------------------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Number%20days%20to%20duration/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc daytomonth(comp)``
       | manual\_ex\_2
     - 🟡
     - ❌
     - ❌
     - type of Me\_2 expected Duration got String
     - `script <examples/time/daytomonth/comp/manual_ex_2/script.vtl>`__ · `input <examples/time/daytomonth/comp/manual_ex_2/input/>`__ · `expected <examples/time/daytomonth/comp/manual_ex_2/expected/>`__ · `notes <examples/time/daytomonth/comp/manual_ex_2/notes.md>`__ · *manual* · erratum
   * - | ``daytomonth(sc)``
       | minimal
     - 🟡
     - ❌
     - ❌
     - type expected Duration got String
     - `script <examples/time/daytomonth/sc/minimal/script.vtl>`__ · `expected <examples/time/daytomonth/sc/minimal/expected/>`__ · `notes <examples/time/daytomonth/sc/minimal/notes.md>`__

🔴 Years and days to number of days ``yeartoday``
------------------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc yeartoday(comp)``
       | manual\_ex\_1
     - 🔴
     - 💥
     - 💥
     - pandas: DataLoadError 0-3-1-6: On Dataset DS\_1 loading: not possible to cast column Me\_1 to Duration. Error found: \"Impossible to cast P2Y230D from type str to Duration.\" Please check loaded file DS\_…
     - `script <examples/time/yeartoday/comp/manual_ex_1/script.vtl>`__ · `input <examples/time/yeartoday/comp/manual_ex_1/input/>`__ · `expected <examples/time/yeartoday/comp/manual_ex_1/expected/>`__ · `notes <examples/time/yeartoday/comp/manual_ex_1/notes.md>`__ · *manual* · erratum
   * - | ``yeartoday(sc)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-5-6: Impossible to cast P1Y20D from type String to Duration. Please check transformation with output Dataset DS\_r
     - `script <examples/time/yeartoday/sc/minimal/script.vtl>`__ · `expected <examples/time/yeartoday/sc/minimal/expected/>`__ · `notes <examples/time/yeartoday/sc/minimal/notes.md>`__

🔴 Months and days to number of days ``monthtoday``
----------------------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``calc monthtoday(comp)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - pandas: DataLoadError 0-3-1-6: On Dataset DS\_1 loading: not possible to cast column Me\_1 to Duration. Error found: \"Impossible to cast P2M5D from type str to Duration.\" Please check loaded file DS\_r;…
     - `script <examples/time/monthtoday/comp/minimal/script.vtl>`__ · `input <examples/time/monthtoday/comp/minimal/input/>`__ · `expected <examples/time/monthtoday/comp/minimal/expected/>`__ · `notes <examples/time/monthtoday/comp/minimal/notes.md>`__
   * - | ``monthtoday(sc)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-5-6: Impossible to cast P3M10D from type String to Duration. Please check transformation with output Dataset DS\_r
     - `script <examples/time/monthtoday/sc/minimal/script.vtl>`__ · `expected <examples/time/monthtoday/sc/minimal/expected/>`__ · `notes <examples/time/monthtoday/sc/minimal/notes.md>`__
