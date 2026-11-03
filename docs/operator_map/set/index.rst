==========================
Set operators
==========================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 4 of 4 operators all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (9c8976e7) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["Set operators"]:::block
       set__union["Union · union<br/>🟢2"]:::ok
       set__intersect["Intersection · intersect<br/>🟢2"]:::ok
       set__setdiff["Set difference · setdiff<br/>🟢1"]:::ok
       set__symdiff["Symmetric difference · symdiff<br/>🟢1"]:::ok
       set__union --- B
       set__intersect --- B
       B --- set__setdiff
       B --- set__symdiff
       click set__union href "union.html"
       click set__intersect href "intersect.html"
       click set__setdiff href "setdiff.html"
       click set__symdiff href "symdiff.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Union <union>` ``union``
     - 🟢
     - 2
   * - :doc:`Intersection <intersect>` ``intersect``
     - 🟢
     - 2
   * - :doc:`Set difference <setdiff>` ``setdiff``
     - 🟢
     - 1
   * - :doc:`Symmetric difference <symdiff>` ``symdiff``
     - 🟢
     - 1

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - any scalar
     - any type
   * - :doc:`Union <union>`
     - in
     - out
   * - :doc:`Intersection <intersect>`
     - in
     - out
   * - :doc:`Set difference <setdiff>`
     - in
     - out
   * - :doc:`Symmetric difference <symdiff>`
     - in
     - out

.. rubric:: Forms that are not green

Every form is green.

.. toctree::
   :maxdepth: 1
   :hidden:

   union
   intersect
   setdiff
   symdiff
