================================
Clause operators
================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 6 of 9 operators all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (52e8d54f) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["Clause operators"]:::block
       clause__calc["Calculation of a Component · calc<br/>🟢5 🟡1"]:::partial
       clause__aggr["Aggregation · aggr<br/>🟢8 🟡1"]:::partial
       clause__filter["Filtering Data Points · filter<br/>🟢1"]:::ok
       clause__keep["Maintaining Components · keep<br/>🟢1"]:::ok
       clause__drop["Removal of Components · drop<br/>🟢1"]:::ok
       clause__rename["Change of Component name · rename<br/>🟢2"]:::ok
       clause__pivot["Pivoting · pivot<br/>🔴1"]:::missing
       clause__unpivot["Unpivoting · unpivot<br/>🟢1"]:::ok
       clause__sub["Subspace · sub<br/>🟢2"]:::ok
       clause__calc --- B
       clause__aggr --- B
       clause__filter --- B
       clause__keep --- B
       clause__drop --- B
       B --- clause__rename
       B --- clause__pivot
       B --- clause__unpivot
       B --- clause__sub
       click clause__calc href "calc.html"
       click clause__aggr href "aggr.html"
       click clause__filter href "filter.html"
       click clause__keep href "keep.html"
       click clause__drop href "drop.html"
       click clause__rename href "rename.html"
       click clause__pivot href "pivot.html"
       click clause__unpivot href "unpivot.html"
       click clause__sub href "sub.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Calculation of a Component <calc>` ``calc``
     - 🟡
     - 6
   * - :doc:`Aggregation <aggr>` ``aggr``
     - 🟡
     - 9
   * - :doc:`Filtering Data Points <filter>` ``filter``
     - 🟢
     - 1
   * - :doc:`Maintaining Components <keep>` ``keep``
     - 🟢
     - 1
   * - :doc:`Removal of Components <drop>` ``drop``
     - 🟢
     - 1
   * - :doc:`Change of Component name <rename>` ``rename``
     - 🟢
     - 2
   * - :doc:`Pivoting <pivot>` ``pivot``
     - 🔴
     - 1
   * - :doc:`Unpivoting <unpivot>` ``unpivot``
     - 🟢
     - 1
   * - :doc:`Subspace <sub>` ``sub``
     - 🟢
     - 2

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - boolean
     - any scalar
     - any type
   * - :doc:`Calculation of a Component <calc>`
     -
     - in
     - out
   * - :doc:`Aggregation <aggr>`
     - in
     - in
     - out
   * - :doc:`Filtering Data Points <filter>`
     - in
     - in
     - out
   * - :doc:`Maintaining Components <keep>`
     -
     - in
     - out
   * - :doc:`Removal of Components <drop>`
     -
     - in
     - out
   * - :doc:`Change of Component name <rename>`
     -
     - in
     - out
   * - :doc:`Pivoting <pivot>`
     -
     - in
     - out
   * - :doc:`Unpivoting <unpivot>`
     -
     - in
     - out
   * - :doc:`Subspace <sub>`
     -
     - in
     - out

.. rubric:: Forms that are not green

Every form that is neither 🟢 nor 🛡️, with the example that shows its status: the first that is neither, or the first that diverges for a 🔵 form.

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
   * - :doc:`Calculation of a Component <calc>`
     - `DS[calc Me := expr] <calc.html#form-measure-default>`__
     - 🟡
     - `Existing attribute keeps role <calc.html#ex-measure-default-existing-attribute-keeps-role>`__
     - ❌
     - ❌
     - role of At\_1 expected Attribute got Measure
   * - :doc:`Aggregation <aggr>`
     - `DS[aggr At := agg(At) group by Id] <aggr.html#form-role-omitted-attribute>`__
     - 🟡
     - `Existing attribute keeps role <aggr.html#ex-role-omitted-attribute-existing-attribute-keeps-role>`__
     - ❌
     - ❌
     - role of At\_1 expected Attribute got Measure
   * - :doc:`Pivoting <pivot>`
     - `DS[pivot Id, Me] <pivot.html#form-basic>`__
     - 🔴
     - `Manual example 1 <pivot.html#ex-basic-manual-ex-1>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: pivot clause. Please check transformation with output Dataset DS\_r

.. toctree::
   :maxdepth: 1
   :hidden:

   calc
   aggr
   filter
   keep
   drop
   rename
   pivot
   unpivot
   sub
