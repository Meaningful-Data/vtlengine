==========================================
Conditional operators
==========================================

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
       B["Conditional operators"]:::block
       conditional__if_then_else["if-then-else · if then else<br/>🟢5"]:::ok
       conditional__case["Case · case when<br/>🟢7"]:::ok
       conditional__nvl["Nvl · nvl<br/>🟢7"]:::ok
       conditional__if_then_else --- B
       conditional__case --- B
       B --- conditional__nvl
       click conditional__if_then_else href "if_then_else.html"
       click conditional__case href "case.html"
       click conditional__nvl href "nvl.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`if-then-else <if_then_else>` ``if then else``
     - 🟢
     - 5
   * - :doc:`Case <case>` ``case when``
     - 🟢
     - 7
   * - :doc:`Nvl <nvl>` ``nvl``
     - 🟢
     - 7

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - boolean
     - any scalar
     - any type
   * - :doc:`if-then-else <if_then_else>`
     - in
     - in
     - out
   * - :doc:`Case <case>`
     - in
     - in
     - out
   * - :doc:`Nvl <nvl>`
     -
     - in
     - out

.. rubric:: Forms that are not green

Every form is 🟢 or 🛡️.

.. toctree::
   :maxdepth: 1
   :hidden:

   if_then_else
   case
   nvl
