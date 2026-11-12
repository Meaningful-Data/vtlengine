================================================
Hierarchical aggregation
================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 1 of 1 operator all ok</span> · <span class="om-gen">

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
       B["Hierarchical aggregation"]:::block
       hierarchy__hierarchy["Hierarchical roll-up · hierarchy 📌<br/>🟢38"]:::ok
       hierarchy__hierarchy --- B
       click hierarchy__hierarchy href "hierarchy.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Hierarchical roll-up <hierarchy>` ``hierarchy``
     - 🟢
     - 38

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - hierarchical ruleset
     - any type
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - in
     - in
     - out

.. rubric:: Forms that are not green

Every form is 🟢 or 🛡️.

.. toctree::
   :maxdepth: 1
   :hidden:

   hierarchy
