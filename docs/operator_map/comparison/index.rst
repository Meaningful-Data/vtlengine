========================================
Comparison operators
========================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 12 of 12 operators all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (d067399d) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["Comparison operators"]:::block
       comparison__equal["Equal to · =<br/>🟢7"]:::ok
       comparison__not_equal["Not equal to · #60;#62;<br/>🟢7"]:::ok
       comparison__greater["Greater than · #62;<br/>🟢7"]:::ok
       comparison__greater_equal["Greater than or equal to · #62;=<br/>🟢7"]:::ok
       comparison__less["Less than · #60;<br/>🟢7"]:::ok
       comparison__less_equal["Less than or equal to · #60;=<br/>🟢7"]:::ok
       comparison__between["Between · between<br/>🟢4"]:::ok
       comparison__in["Element of · in<br/>🟢5"]:::ok
       comparison__not_in["Not element of · not_in<br/>🟢5"]:::ok
       comparison__match_characters["Match characters · match_characters<br/>🟢4"]:::ok
       comparison__isnull["Is null · isnull<br/>🟢3"]:::ok
       comparison__exists_in["Exists in · exists_in<br/>🟢4"]:::ok
       comparison__equal --- B
       comparison__not_equal --- B
       comparison__greater --- B
       comparison__greater_equal --- B
       comparison__less --- B
       comparison__less_equal --- B
       B --- comparison__between
       B --- comparison__in
       B --- comparison__not_in
       B --- comparison__match_characters
       B --- comparison__isnull
       B --- comparison__exists_in
       click comparison__equal href "equal.html"
       click comparison__not_equal href "not_equal.html"
       click comparison__greater href "greater.html"
       click comparison__greater_equal href "greater_equal.html"
       click comparison__less href "less.html"
       click comparison__less_equal href "less_equal.html"
       click comparison__between href "between.html"
       click comparison__in href "in.html"
       click comparison__not_in href "not_in.html"
       click comparison__match_characters href "match_characters.html"
       click comparison__isnull href "isnull.html"
       click comparison__exists_in href "exists_in.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Equal to <equal>` ``=``
     - 🟢
     - 7
   * - :doc:`Not equal to <not_equal>` ``<>``
     - 🟢
     - 7
   * - :doc:`Greater than <greater>` ``>``
     - 🟢
     - 7
   * - :doc:`Greater than or equal to <greater_equal>` ``>=``
     - 🟢
     - 7
   * - :doc:`Less than <less>` ``<``
     - 🟢
     - 7
   * - :doc:`Less than or equal to <less_equal>` ``<=``
     - 🟢
     - 7
   * - :doc:`Between <between>` ``between``
     - 🟢
     - 4
   * - :doc:`Element of <in>` ``in``
     - 🟢
     - 5
   * - :doc:`Not element of <not_in>` ``not_in``
     - 🟢
     - 5
   * - :doc:`Match characters <match_characters>` ``match_characters``
     - 🟢
     - 4
   * - :doc:`Is null <isnull>` ``isnull``
     - 🟢
     - 3
   * - :doc:`Exists in <exists_in>` ``exists_in``
     - 🟢
     - 4

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - string
     - boolean
     - set
     - value domain
     - any scalar
   * - :doc:`Equal to <equal>`
     -
     - out
     -
     -
     - in
   * - :doc:`Not equal to <not_equal>`
     -
     - out
     -
     -
     - in
   * - :doc:`Greater than <greater>`
     -
     - out
     -
     -
     - in
   * - :doc:`Greater than or equal to <greater_equal>`
     -
     - out
     -
     -
     - in
   * - :doc:`Less than <less>`
     -
     - out
     -
     -
     - in
   * - :doc:`Less than or equal to <less_equal>`
     -
     - out
     -
     -
     - in
   * - :doc:`Between <between>`
     -
     - out
     -
     -
     - in
   * - :doc:`Element of <in>`
     -
     - out
     - in
     - in
     - in
   * - :doc:`Not element of <not_in>`
     -
     - out
     - in
     - in
     - in
   * - :doc:`Match characters <match_characters>`
     - in
     - out
     -
     -
     -
   * - :doc:`Is null <isnull>`
     -
     - out
     -
     -
     - in
   * - :doc:`Exists in <exists_in>`
     -
     - out
     -
     -
     - in

.. rubric:: Forms that are not green

Every form is green.

.. toctree::
   :maxdepth: 1
   :hidden:

   equal
   not_equal
   greater
   greater_equal
   less
   less_equal
   between
   in
   not_in
   match_characters
   isnull
   exists_in
