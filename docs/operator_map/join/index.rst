============================
Join operators
============================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 0 of 4 operators all ok</span> · <span class="om-gen">vtlengine 1.9.6rc2 (e84ced22) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       join__inner_join["Inner Join · inner_join<br/>🟢19 🟡3"]:::partial
       join__left_join["Left Join · left_join<br/>🟢22 🟡1"]:::partial
       join__full_join["Full Join · full_join<br/>🟢21 🔴2"]:::partial
       join__cross_join["Cross Join · cross_join<br/>🟢20 🟡1"]:::partial
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
     - 🟡
     - 22
   * - :doc:`Left Join <left_join>` ``left_join``
     - 🟡
     - 23
   * - :doc:`Full Join <full_join>` ``full_join``
     - 🟡
     - 23
   * - :doc:`Cross Join <cross_join>` ``cross_join``
     - 🟡
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
   * - :doc:`Inner Join <inner_join>`
     - `inner\_join(DS, DS) <inner_join.html#form-bare>`__
     - 🟡
     - `One shared identifier <inner_join.html#ex-bare-one-shared-identifier>`__
     - 🛡 1-1-13-11
     - 🛡 1-1-13-11
     - SemanticError 1-1-13-11: At op inner\_join: Invalid subcase A, DS\_1 should be a superset but Id\_3 not found. Please check transformation with output Dataset DS\_r
   * - :doc:`Inner Join <inner_join>`
     - `inner\_join(DS, DS using Id) <inner_join.html#form-using>`__
     - 🟡
     - `Subset of identifiers <inner_join.html#ex-using-subset-of-identifiers>`__
     - 🛡 1-1-13-4
     - 🛡 1-1-13-4
     - SemanticError 1-1-13-4: At op inner\_join: Using clause, using=[\'Id\_1\'], does not define all the Identifiers, of non reference Dataset d2. Please check transformation with output Dataset DS\_r
   * - :doc:`Inner Join <inner_join>`
     - `inner\_join(DS, DS, DS) <inner_join.html#form-nary>`__
     - 🟡
     - `Manual example 5 <inner_join.html#ex-nary-manual-ex-5>`__
     - 🛡 1-1-13-11
     - 🛡 1-1-13-11
     - SemanticError 1-1-13-11: At op inner\_join: Invalid subcase A, DS\_5 should be a superset but Id\_1 not found. Please check transformation with output Dataset IBSC
   * - :doc:`Left Join <left_join>`
     - `left\_join(DS, DS using Id, nvl(Id, c)) <left_join.html#form-using-nvl>`__
     - 🟡
     - `Nvl for extra identifier <left_join.html#ex-using-nvl-nvl-for-extra-identifier>`__
     - 🛡 1-1-13-4
     - 🛡 1-1-13-4
     - SemanticError 1-1-13-4: At op left\_join: Using clause, using=[\'Id\_1\'], does not define all the Identifiers, of non reference Dataset DS\_2. Please check transformation with output Dataset DS\_r
   * - :doc:`Full Join <full_join>`
     - `full\_join(DS, DS using Id) <full_join.html#form-using>`__
     - 🔴
     - `All common identifiers <full_join.html#ex-using-all-common-identifiers>`__
     - 🛡 1-1-13-8
     - 🛡 1-1-13-8
     - SemanticError 1-1-13-8: At op full\_join: No available using clause. Please check transformation with output Dataset DS\_r
   * - :doc:`Full Join <full_join>`
     - `full\_join(DS, DS using Id, nvl(Id, c)) <full_join.html#form-using-nvl>`__
     - 🔴
     - `Nvl for extra identifier <full_join.html#ex-using-nvl-nvl-for-extra-identifier>`__
     - 🛡 1-1-13-8
     - 🛡 1-1-13-8
     - SemanticError 1-1-13-8: At op full\_join: No available using clause. Please check transformation with output Dataset DS\_r
   * - :doc:`Cross Join <cross_join>`
     - `cross\_join(DS, DS) <cross_join.html#form-bare>`__
     - 🟡
     - `Expression without alias <cross_join.html#ex-bare-expression-without-alias>`__
     - ❌
     - ❌
     - expected error 1-1-13-2, got 1-1-13-9

.. toctree::
   :maxdepth: 1
   :hidden:

   inner_join
   left_join
   full_join
   cross_join
