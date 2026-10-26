============================
Time operators
============================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 11 of 17 operators all ok</span> · <span class="om-gen">vtlengine 1.9.6rc2 (9dc0a08a) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

.. raw:: html

   <div class="om-legend"><span class="om-chip ok">🟢 all ok</span><span class="om-chip partial">🟡 partially implemented</span><span class="om-chip refused">🛡️ refused with an error</span><span class="om-chip missing">🔴 not implemented or crash</span><span class="om-chip diverge">🔵 engines differ</span> <span class="om-key">✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash · 📌 team decision</span></div>

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
       B["Time operators"]:::block
       time__period_indicator["Period indicator · period_indicator<br/>🟢3 🔴1"]:::partial
       time__fill_time_series["Fill time series · fill_time_series<br/>🟢8"]:::ok
       time__flow_to_stock["Flow to stock · flow_to_stock<br/>🟢3"]:::ok
       time__stock_to_flow["Stock to flow · stock_to_flow<br/>🟢3"]:::ok
       time__timeshift["Time shift · timeshift<br/>🟢3"]:::ok
       time__time_agg["Time aggregation · time_agg<br/>🟢7 🛡️1"]:::partial
       time__current_date["Actual time · current_date<br/>🟢1"]:::ok
       time__datediff["Days between two dates · datediff<br/>🟢4"]:::ok
       time__dateadd["Add time unit to date · dateadd<br/>🟢4"]:::ok
       time__getyear["Extract the year · getyear<br/>🟢2"]:::ok
       time__getmonth["Extract the month · getmonth<br/>🟢2"]:::ok
       time__dayofmonth["Extract the day of the month · dayofmonth<br/>🟢2"]:::ok
       time__dayofyear["Extract the day of the year · dayofyear<br/>🟢2"]:::ok
       time__daytoyear["Number of days to years and days · daytoyear<br/>🟡2"]:::partial
       time__daytomonth["Number of days to months and days · daytomonth<br/>🟡2"]:::partial
       time__yeartoday["Years and days to number of days · yeartoday<br/>🛡️2"]:::refused
       time__monthtoday["Months and days to number of days · monthtoday<br/>🛡️2"]:::refused
       time__period_indicator --- B
       time__fill_time_series --- B
       time__flow_to_stock --- B
       time__stock_to_flow --- B
       time__timeshift --- B
       time__time_agg --- B
       time__current_date --- B
       time__datediff --- B
       time__dateadd --- B
       B --- time__getyear
       B --- time__getmonth
       B --- time__dayofmonth
       B --- time__dayofyear
       B --- time__daytoyear
       B --- time__daytomonth
       B --- time__yeartoday
       B --- time__monthtoday
       click time__period_indicator href "period_indicator.html"
       click time__fill_time_series href "fill_time_series.html"
       click time__flow_to_stock href "flow_to_stock.html"
       click time__stock_to_flow href "stock_to_flow.html"
       click time__timeshift href "timeshift.html"
       click time__time_agg href "time_agg.html"
       click time__current_date href "current_date.html"
       click time__datediff href "datediff.html"
       click time__dateadd href "dateadd.html"
       click time__getyear href "getyear.html"
       click time__getmonth href "getmonth.html"
       click time__dayofmonth href "dayofmonth.html"
       click time__dayofyear href "dayofyear.html"
       click time__daytoyear href "daytoyear.html"
       click time__daytomonth href "daytomonth.html"
       click time__yeartoday href "yeartoday.html"
       click time__monthtoday href "monthtoday.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Period indicator <period_indicator>` ``period_indicator``
     - 🟡
     - 4
   * - :doc:`Fill time series <fill_time_series>` ``fill_time_series``
     - 🟢
     - 8
   * - :doc:`Flow to stock <flow_to_stock>` ``flow_to_stock``
     - 🟢
     - 3
   * - :doc:`Stock to flow <stock_to_flow>` ``stock_to_flow``
     - 🟢
     - 3
   * - :doc:`Time shift <timeshift>` ``timeshift``
     - 🟢
     - 3
   * - :doc:`Time aggregation <time_agg>` ``time_agg``
     - 🟡
     - 8
   * - :doc:`Actual time <current_date>` ``current_date``
     - 🟢
     - 1
   * - :doc:`Days between two dates <datediff>` ``datediff``
     - 🟢
     - 4
   * - :doc:`Add time unit to date <dateadd>` ``dateadd``
     - 🟢
     - 4
   * - :doc:`Extract the year <getyear>` ``getyear``
     - 🟢
     - 2
   * - :doc:`Extract the month <getmonth>` ``getmonth``
     - 🟢
     - 2
   * - :doc:`Extract the day of the month <dayofmonth>` ``dayofmonth``
     - 🟢
     - 2
   * - :doc:`Extract the day of the year <dayofyear>` ``dayofyear``
     - 🟢
     - 2
   * - :doc:`Number of days to years and days <daytoyear>` ``daytoyear``
     - 🟡
     - 2
   * - :doc:`Number of days to months and days <daytomonth>` ``daytomonth``
     - 🟡
     - 2
   * - :doc:`Years and days to number of days <yeartoday>` ``yeartoday``
     - 🛡️
     - 2
   * - :doc:`Months and days to number of days <monthtoday>` ``monthtoday``
     - 🛡️
     - 2

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - integer
     - time
     - date
     - time_period
     - duration
     - any type
   * - :doc:`Period indicator <period_indicator>`
     -
     -
     -
     -
     - in
     - out
     -
   * - :doc:`Fill time series <fill_time_series>`
     -
     -
     - in · out
     - in · out
     - in · out
     -
     - out
   * - :doc:`Flow to stock <flow_to_stock>`
     - in · out
     -
     - in · out
     - in · out
     - in · out
     -
     -
   * - :doc:`Stock to flow <stock_to_flow>`
     - in · out
     -
     - in · out
     - in · out
     - in · out
     -
     -
   * - :doc:`Time shift <timeshift>`
     -
     - in
     - in · out
     - in · out
     - in · out
     -
     - out
   * - :doc:`Time aggregation <time_agg>`
     -
     -
     - in · out
     - in · out
     - in · out
     - in
     -
   * - :doc:`Actual time <current_date>`
     -
     -
     -
     - out
     -
     -
     -
   * - :doc:`Days between two dates <datediff>`
     -
     - out
     -
     - in
     - in
     -
     -
   * - :doc:`Add time unit to date <dateadd>`
     -
     - in
     - in · out
     - in · out
     - in
     - in
     - out
   * - :doc:`Extract the year <getyear>`
     -
     - out
     -
     - in
     - in
     -
     -
   * - :doc:`Extract the month <getmonth>`
     -
     - out
     -
     - in
     - in
     -
     -
   * - :doc:`Extract the day of the month <dayofmonth>`
     -
     - out
     -
     - in
     - in
     -
     -
   * - :doc:`Extract the day of the year <dayofyear>`
     -
     - out
     -
     - in
     - in
     -
     -
   * - :doc:`Number of days to years and days <daytoyear>`
     -
     - in
     -
     -
     -
     - out
     -
   * - :doc:`Number of days to months and days <daytomonth>`
     -
     - in
     -
     -
     -
     - out
     -
   * - :doc:`Years and days to number of days <yeartoday>`
     -
     - out
     -
     -
     -
     - in
     -
   * - :doc:`Months and days to number of days <monthtoday>`
     -
     - out
     -
     -
     -
     - in
     -

.. rubric:: Forms that are not green

Every form that is not 🟢, with the example that shows its status: the first that is not green, or the first that diverges for a 🔵 form.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Form
     - Status
     - Example
     - pandas
     - DuckDB
     - Detail
   * - :doc:`Period indicator <period_indicator>`
     - `filter period\_indicator() <period_indicator.html#form-in-filter-no-operand>`__
     - 🔴
     - `Keeps the quarterly data point <period_indicator.html#ex-in-filter-no-operand-keeps-the-quarterly-data-point>`__
     - 💥 IndexError
     - 💥 IndexError
     - IndexError: list index out of range
   * - :doc:`Time aggregation <time_agg>`
     - `time\_agg(p, sc) on date <time_agg.html#form-sc-date>`__
     - 🛡️
     - `Without first or last <time_agg.html#ex-sc-date-without-first-or-last>`__
     - 🛡 1-1-19-11
     - 🛡 1-1-19-11
     - SemanticError 1-1-19-11: At op time\_agg: If used over a Date, first/last parameter must be declared. Please check transformation with output Dataset DS\_r
   * - :doc:`Number of days to years and days <daytoyear>`
     - `calc daytoyear(comp) <daytoyear.html#form-comp>`__
     - 🟡
     - `Manual example 1 <daytoyear.html#ex-comp-manual-ex-1>`__
     - ❌
     - ❌
     - type of Me\_2 expected Duration got String
   * - :doc:`Number of days to years and days <daytoyear>`
     - `daytoyear(sc) <daytoyear.html#form-sc>`__
     - 🟡
     - `Years of 365 days <daytoyear.html#ex-sc-years-of-365-days>`__
     - ❌
     - ❌
     - type expected Duration got String
   * - :doc:`Number of days to months and days <daytomonth>`
     - `calc daytomonth(comp) <daytomonth.html#form-comp>`__
     - 🟡
     - `Manual example 2 <daytomonth.html#ex-comp-manual-ex-2>`__
     - ❌
     - ❌
     - type of Me\_2 expected Duration got String
   * - :doc:`Number of days to months and days <daytomonth>`
     - `daytomonth(sc) <daytomonth.html#form-sc>`__
     - 🟡
     - `Months of thirty days <daytomonth.html#ex-sc-months-of-thirty-days>`__
     - ❌
     - ❌
     - type expected Duration got String
   * - :doc:`Years and days to number of days <yeartoday>`
     - `calc yeartoday(comp) <yeartoday.html#form-comp>`__
     - 🛡️
     - `Manual example 1 <yeartoday.html#ex-comp-manual-ex-1>`__
     - 🛡 0-3-1-6
     - 🛡 0-3-1-6
     - pandas: DataLoadError 0-3-1-6: On Dataset DS\_1 loading: not possible to cast column Me\_1 to Duration. Error found: \"Impossible to cast P2Y230D from type str to Duration.\" Please check loaded file DS\_…
   * - :doc:`Years and days to number of days <yeartoday>`
     - `yeartoday(sc) <yeartoday.html#form-sc>`__
     - 🛡️
     - `One year and twenty days <yeartoday.html#ex-sc-one-year-and-twenty-days>`__
     - 🛡 1-1-5-6
     - 🛡 1-1-5-6
     - SemanticError 1-1-5-6: Impossible to cast P1Y20D from type String to Duration. Please check transformation with output Dataset DS\_r
   * - :doc:`Months and days to number of days <monthtoday>`
     - `calc monthtoday(comp) <monthtoday.html#form-comp>`__
     - 🛡️
     - `Iso month and day durations <monthtoday.html#ex-comp-iso-month-and-day-durations>`__
     - 🛡 0-3-1-6
     - 🛡 0-3-1-6
     - pandas: DataLoadError 0-3-1-6: On Dataset DS\_1 loading: not possible to cast column Me\_1 to Duration. Error found: \"Impossible to cast P2M5D from type str to Duration.\" Please check loaded file DS\_r;…
   * - :doc:`Months and days to number of days <monthtoday>`
     - `monthtoday(sc) <monthtoday.html#form-sc>`__
     - 🛡️
     - `Three months and ten days <monthtoday.html#ex-sc-three-months-and-ten-days>`__
     - 🛡 1-1-5-6
     - 🛡 1-1-5-6
     - SemanticError 1-1-5-6: Impossible to cast P3M10D from type String to Duration. Please check transformation with output Dataset DS\_r

.. toctree::
   :maxdepth: 1
   :hidden:

   period_indicator
   fill_time_series
   flow_to_stock
   stock_to_flow
   timeshift
   time_agg
   current_date
   datediff
   dateadd
   getyear
   getmonth
   dayofmonth
   dayofyear
   daytoyear
   daytomonth
   yeartoday
   monthtoday
