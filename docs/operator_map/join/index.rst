============================
Join operators
============================

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
       B["Join operators"]:::block
       join__inner_join["Inner Join · inner_join 📌<br/>🟢22"]:::ok
       join__left_join["Left Join · left_join 📌<br/>🟢23"]:::ok
       join__full_join["Full Join · full_join 📌<br/>🟢23"]:::ok
       join__cross_join["Cross Join · cross_join 📌<br/>🟢21"]:::ok
       join__inner_join --- B
       join__left_join --- B
       B --- join__full_join
       B --- join__cross_join
       click join__inner_join href "inner_join.html"
       click join__left_join href "left_join.html"
       click join__full_join href "full_join.html"
       click join__cross_join href "cross_join.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Inner Join <inner_join>` ``inner_join``
     - 🟢
     - 22
   * - :doc:`Left Join <left_join>` ``left_join``
     - 🟢
     - 23
   * - :doc:`Full Join <full_join>` ``full_join``
     - 🟢
     - 23
   * - :doc:`Cross Join <cross_join>` ``cross_join``
     - 🟢
     - 21

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - boolean
     - any scalar
     - any type
   * - :doc:`Inner Join <inner_join>`
     - in
     - in
     - out
   * - :doc:`Left Join <left_join>`
     - in
     - in
     - out
   * - :doc:`Full Join <full_join>`
     - in
     - in
     - out
   * - :doc:`Cross Join <cross_join>`
     - in
     - in
     - out

.. rubric:: Forms that are not green

Every form is 🟢 or 🛡️.

.. toctree::
   :maxdepth: 1
   :hidden:

   inner_join
   left_join
   full_join
   cross_join
