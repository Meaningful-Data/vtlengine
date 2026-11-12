==================================================
Data validation operators
==================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 3 of 3 operators all ok</span> · <span class="om-gen">

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
       B["Data validation operators"]:::block
       validation__check_datapoint["Check datapoint · check_datapoint<br/>🟢7"]:::ok
       validation__check_hierarchy["Check hierarchy · check_hierarchy<br/>🟢40"]:::ok
       validation__check["Check · check<br/>🟢6"]:::ok
       validation__check_datapoint --- B
       validation__check_hierarchy --- B
       B --- validation__check
       click validation__check_datapoint href "check_datapoint.html"
       click validation__check_hierarchy href "check_hierarchy.html"
       click validation__check href "check.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Check datapoint <check_datapoint>` ``check_datapoint``
     - 🟢
     - 7
   * - :doc:`Check hierarchy <check_hierarchy>` ``check_hierarchy``
     - 🟢
     - 40
   * - :doc:`Check <check>` ``check``
     - 🟢
     - 6

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - integer
     - string
     - boolean
     - datapoint ruleset
     - hierarchical ruleset
     - any scalar
     - any type
   * - :doc:`Check datapoint <check_datapoint>`
     -
     - out
     - out
     - out
     - in
     -
     - in
     - out
   * - :doc:`Check hierarchy <check_hierarchy>`
     - in · out
     - out
     - out
     - out
     -
     - in
     -
     -
   * - :doc:`Check <check>`
     - in · out
     - in · out
     - in · out
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

   check_datapoint
   check_hierarchy
   check
