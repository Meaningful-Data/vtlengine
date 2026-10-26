==================================
Boolean operators
==================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 4 of 4 operators all ok</span> · <span class="om-gen">vtlengine 1.9.6rc2 (140c5c19) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["Boolean operators"]:::block
       boolean__and["Logical conjunction · and<br/>🟢7"]:::ok
       boolean__or["Logical disjunction · or<br/>🟢7"]:::ok
       boolean__xor["Exclusive disjunction · xor<br/>🟢7"]:::ok
       boolean__not["Logical negation · not<br/>🟢3"]:::ok
       boolean__and --- B
       boolean__or --- B
       B --- boolean__xor
       B --- boolean__not
       click boolean__and href "and.html"
       click boolean__or href "or.html"
       click boolean__xor href "xor.html"
       click boolean__not href "not.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Logical conjunction <and>` ``and``
     - 🟢
     - 7
   * - :doc:`Logical disjunction <or>` ``or``
     - 🟢
     - 7
   * - :doc:`Exclusive disjunction <xor>` ``xor``
     - 🟢
     - 7
   * - :doc:`Logical negation <not>` ``not``
     - 🟢
     - 3

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - boolean
   * - :doc:`Logical conjunction <and>`
     - in · out
   * - :doc:`Logical disjunction <or>`
     - in · out
   * - :doc:`Exclusive disjunction <xor>`
     - in · out
   * - :doc:`Logical negation <not>`
     - in · out

.. rubric:: Forms that are not green

Every form is green.

.. toctree::
   :maxdepth: 1
   :hidden:

   and
   or
   xor
   not
