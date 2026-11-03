================================================
Hierarchical aggregation
================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 0 of 1 operator all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (ce4408d3) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       hierarchy__hierarchy["Hierarchical roll-up · hierarchy 📌<br/>🟢26 🟡12"]:::partial
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
     - 🟡
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
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_null dataset computed) <hierarchy.html#form-always-null-dataset-computed>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-null-dataset-computed-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_null dataset all) <hierarchy.html#form-always-null-dataset-all>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-null-dataset-all-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_null rule computed) <hierarchy.html#form-always-null-rule-computed>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-null-rule-computed-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_null rule all) <hierarchy.html#form-always-null-rule-all>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-null-rule-all-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_null rule\_priority computed) <hierarchy.html#form-always-null-rule-priority-computed>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-null-rule-priority-computed-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_null rule\_priority all) <hierarchy.html#form-always-null-rule-priority-all>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-null-rule-priority-all-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_zero dataset computed) <hierarchy.html#form-always-zero-dataset-computed>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-zero-dataset-computed-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_zero dataset all) <hierarchy.html#form-always-zero-dataset-all>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-zero-dataset-all-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_zero rule computed) <hierarchy.html#form-always-zero-rule-computed>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-zero-rule-computed-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_zero rule all) <hierarchy.html#form-always-zero-rule-all>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-zero-rule-all-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_zero rule\_priority computed) <hierarchy.html#form-always-zero-rule-priority-computed>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-zero-rule-priority-computed-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
   * - :doc:`Hierarchical roll-up <hierarchy>`
     - `hierarchy(DS, HR always\_zero rule\_priority all) <hierarchy.html#form-always-zero-rule-priority-all>`__
     - 🟡
     - `Missing null and zero items <hierarchy.html#ex-always-zero-rule-priority-all-missing-null-and-zero-items>`__
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C

.. toctree::
   :maxdepth: 1
   :hidden:

   hierarchy
