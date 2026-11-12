================================================================
Aggregate and Analytic operators
================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 18 of 18 operators all ok</span> · <span class="om-gen">

.. raw:: html
   :file: ../run.html

.. raw:: html

   </span></div>

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
       B["Aggregate and Analytic operators"]:::block
       aggregate__aggregate_invocation["Aggregate invocation · group by<br/>🟢4"]:::ok
       aggregate__analytic_invocation["Analytic invocation · over 📌<br/>🟢11"]:::ok
       aggregate__count["Counting the number of data points · count 📌<br/>🟢9"]:::ok
       aggregate__min["Minimum value · min<br/>🟢8"]:::ok
       aggregate__max["Maximum value · max<br/>🟢8"]:::ok
       aggregate__median["Median value · median<br/>🟢8"]:::ok
       aggregate__sum["Sum · sum<br/>🟢8"]:::ok
       aggregate__avg["Average value · avg<br/>🟢8"]:::ok
       aggregate__stddev_pop["Population standard deviation · stddev_pop<br/>🟢8"]:::ok
       aggregate__stddev_samp["Sample standard deviation · stddev_samp<br/>🟢8"]:::ok
       aggregate__var_pop["Population variance · var_pop<br/>🟢8"]:::ok
       aggregate__var_samp["Sample variance · var_samp<br/>🟢8"]:::ok
       aggregate__first_value["First value · first_value<br/>🟢2"]:::ok
       aggregate__last_value["Last value · last_value<br/>🟢2"]:::ok
       aggregate__lag["Lag · lag<br/>🟢4"]:::ok
       aggregate__lead["Lead · lead<br/>🟢4"]:::ok
       aggregate__rank["Rank · rank<br/>🟢2"]:::ok
       aggregate__ratio_to_report["Ratio to report · ratio_to_report<br/>🟢2"]:::ok
       aggregate__aggregate_invocation --- B
       aggregate__analytic_invocation --- B
       aggregate__count --- B
       aggregate__min --- B
       aggregate__max --- B
       aggregate__median --- B
       aggregate__sum --- B
       aggregate__avg --- B
       aggregate__stddev_pop --- B
       B --- aggregate__stddev_samp
       B --- aggregate__var_pop
       B --- aggregate__var_samp
       B --- aggregate__first_value
       B --- aggregate__last_value
       B --- aggregate__lag
       B --- aggregate__lead
       B --- aggregate__rank
       B --- aggregate__ratio_to_report
       click aggregate__aggregate_invocation href "aggregate_invocation.html"
       click aggregate__analytic_invocation href "analytic_invocation.html"
       click aggregate__count href "count.html"
       click aggregate__min href "min.html"
       click aggregate__max href "max.html"
       click aggregate__median href "median.html"
       click aggregate__sum href "sum.html"
       click aggregate__avg href "avg.html"
       click aggregate__stddev_pop href "stddev_pop.html"
       click aggregate__stddev_samp href "stddev_samp.html"
       click aggregate__var_pop href "var_pop.html"
       click aggregate__var_samp href "var_samp.html"
       click aggregate__first_value href "first_value.html"
       click aggregate__last_value href "last_value.html"
       click aggregate__lag href "lag.html"
       click aggregate__lead href "lead.html"
       click aggregate__rank href "rank.html"
       click aggregate__ratio_to_report href "ratio_to_report.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Aggregate invocation <aggregate_invocation>` ``group by``
     - 🟢
     - 4
   * - :doc:`Analytic invocation <analytic_invocation>` ``over``
     - 🟢
     - 11
   * - :doc:`Counting the number of data points <count>` ``count``
     - 🟢
     - 9
   * - :doc:`Minimum value <min>` ``min``
     - 🟢
     - 8
   * - :doc:`Maximum value <max>` ``max``
     - 🟢
     - 8
   * - :doc:`Median value <median>` ``median``
     - 🟢
     - 8
   * - :doc:`Sum <sum>` ``sum``
     - 🟢
     - 8
   * - :doc:`Average value <avg>` ``avg``
     - 🟢
     - 8
   * - :doc:`Population standard deviation <stddev_pop>` ``stddev_pop``
     - 🟢
     - 8
   * - :doc:`Sample standard deviation <stddev_samp>` ``stddev_samp``
     - 🟢
     - 8
   * - :doc:`Population variance <var_pop>` ``var_pop``
     - 🟢
     - 8
   * - :doc:`Sample variance <var_samp>` ``var_samp``
     - 🟢
     - 8
   * - :doc:`First value <first_value>` ``first_value``
     - 🟢
     - 2
   * - :doc:`Last value <last_value>` ``last_value``
     - 🟢
     - 2
   * - :doc:`Lag <lag>` ``lag``
     - 🟢
     - 4
   * - :doc:`Lead <lead>` ``lead``
     - 🟢
     - 4
   * - :doc:`Rank <rank>` ``rank``
     - 🟢
     - 2
   * - :doc:`Ratio to report <ratio_to_report>` ``ratio_to_report``
     - 🟢
     - 2

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - integer
     - boolean
     - any scalar
     - any type
   * - :doc:`Aggregate invocation <aggregate_invocation>`
     -
     -
     -
     - in
     - out
   * - :doc:`Analytic invocation <analytic_invocation>`
     -
     -
     -
     - in
     - out
   * - :doc:`Counting the number of data points <count>`
     -
     - out
     - in
     - in
     -
   * - :doc:`Minimum value <min>`
     -
     -
     - in
     - in
     - out
   * - :doc:`Maximum value <max>`
     -
     -
     - in
     - in
     - out
   * - :doc:`Median value <median>`
     - in · out
     -
     - in
     -
     -
   * - :doc:`Sum <sum>`
     - in · out
     -
     - in
     -
     -
   * - :doc:`Average value <avg>`
     - in · out
     -
     - in
     -
     -
   * - :doc:`Population standard deviation <stddev_pop>`
     - in · out
     -
     - in
     -
     -
   * - :doc:`Sample standard deviation <stddev_samp>`
     - in · out
     -
     - in
     -
     -
   * - :doc:`Population variance <var_pop>`
     - in · out
     -
     - in
     -
     -
   * - :doc:`Sample variance <var_samp>`
     - in · out
     -
     - in
     -
     -
   * - :doc:`First value <first_value>`
     -
     -
     -
     - in
     - out
   * - :doc:`Last value <last_value>`
     -
     -
     -
     - in
     - out
   * - :doc:`Lag <lag>`
     -
     - in
     -
     - in
     - out
   * - :doc:`Lead <lead>`
     -
     - in
     -
     - in
     - out
   * - :doc:`Rank <rank>`
     -
     - out
     -
     - in
     -
   * - :doc:`Ratio to report <ratio_to_report>`
     - in · out
     -
     -
     -
     -

.. rubric:: Forms that are not green

Every form is 🟢 or 🛡️.

.. toctree::
   :maxdepth: 1
   :hidden:

   aggregate_invocation
   analytic_invocation
   count
   min
   max
   median
   sum
   avg
   stddev_pop
   stddev_samp
   var_pop
   var_samp
   first_value
   last_value
   lag
   lead
   rank
   ratio_to_report
