========================================
Definitions (VTL-DL)
========================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 4 of 4 operators all ok</span> · <span class="om-gen">

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
       B["Definitions (VTL-DL)"]:::block
       definitions__define_operator["define operator 📌<br/>🟢13"]:::ok
       definitions__define_datapoint_ruleset["define datapoint ruleset<br/>🟢5"]:::ok
       definitions__define_hierarchical_ruleset["define hierarchical ruleset<br/>🟢7"]:::ok
       definitions__define_viral_propagation["define viral propagation<br/>🟢4"]:::ok
       definitions__define_operator --- B
       definitions__define_datapoint_ruleset --- B
       B --- definitions__define_hierarchical_ruleset
       B --- definitions__define_viral_propagation
       click definitions__define_operator href "define_operator.html"
       click definitions__define_datapoint_ruleset href "define_datapoint_ruleset.html"
       click definitions__define_hierarchical_ruleset href "define_hierarchical_ruleset.html"
       click definitions__define_viral_propagation href "define_viral_propagation.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`define operator <define_operator>`
     - 🟢
     - 13
   * - :doc:`define datapoint ruleset <define_datapoint_ruleset>`
     - 🟢
     - 5
   * - :doc:`define hierarchical ruleset <define_hierarchical_ruleset>`
     - 🟢
     - 7
   * - :doc:`define viral propagation <define_viral_propagation>`
     - 🟢
     - 4

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - integer
     - string
     - boolean
     - time
     - date
     - time_period
     - duration
     - set
     - value domain
     - datapoint ruleset
     - hierarchical ruleset
     - viral propagation
     - operator
     - any scalar
   * - :doc:`define operator <define_operator>`
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     -
     - out
     - in
   * - :doc:`define datapoint ruleset <define_datapoint_ruleset>`
     -
     - in
     - in
     - in
     -
     -
     -
     -
     -
     - in
     - out
     -
     -
     -
     -
   * - :doc:`define hierarchical ruleset <define_hierarchical_ruleset>`
     -
     - in
     - in
     - in
     -
     -
     -
     -
     -
     - in
     -
     - out
     -
     -
     -
   * - :doc:`define viral propagation <define_viral_propagation>`
     - in
     -
     -
     -
     -
     -
     -
     -
     -
     - in
     -
     -
     - out
     -
     - in

.. rubric:: Forms that are not green

Every form is 🟢 or 🛡️.

.. toctree::
   :maxdepth: 1
   :hidden:

   define_operator
   define_datapoint_ruleset
   define_hierarchical_ruleset
   define_viral_propagation
