================
Overview
================

.. raw:: html

   <div class="om-legend"><span class="om-chip ok">🟢 all ok</span><span class="om-chip partial">🟡 partially implemented</span><span class="om-chip refused">🛡️ refused with an error</span><span class="om-chip missing">🔴 not implemented or crash</span><span class="om-chip diverge">🔵 engines differ</span> <span class="om-key">✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash · 📌 team decision</span></div>

.. rubric:: Overview diagram

A node for each block, with how many operators it has, and how many of them are in each status, drawn small: the ⛶ button opens it in full screen. Where the viewer supports links in diagrams, a block opens its page. The tables below are the reliable way to every page.

.. raw:: html

   <div class="om-diagram om-small">

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
       R["VTL 2.2<br/>Reference Manual"]:::block
       chapter__general["General purpose operators<br/>7 operators · 🟢6 🟡1"]:::partial
       chapter__join["Join operators<br/>4 operators · 🟢4"]:::ok
       chapter__string["String operators<br/>11 operators · 🟢11"]:::ok
       chapter__numeric["Numeric operators<br/>18 operators · 🟢18"]:::ok
       chapter__comparison["Comparison operators<br/>12 operators · 🟢12"]:::ok
       chapter__boolean["Boolean operators<br/>4 operators · 🟢4"]:::ok
       chapter__time["Time operators<br/>17 operators · 🟢17"]:::ok
       chapter__set["Set operators<br/>4 operators · 🟢4"]:::ok
       chapter__hierarchy["Hierarchical aggregation<br/>1 operator · 🟢1"]:::ok
       chapter__aggregate["Aggregate and Analytic operators<br/>18 operators · 🟢18"]:::ok
       chapter__validation["Data validation operators<br/>3 operators · 🟢3"]:::ok
       chapter__conditional["Conditional operators<br/>3 operators · 🟢3"]:::ok
       chapter__clause["Clause operators<br/>9 operators · 🟢9"]:::ok
       chapter__definitions["Definitions (VTL-DL)<br/>4 operators · 🟢4"]:::ok
       chapter__general --- R
       chapter__join --- R
       chapter__string --- R
       chapter__numeric --- R
       chapter__comparison --- R
       chapter__boolean --- R
       chapter__time --- R
       R --- chapter__set
       R --- chapter__hierarchy
       R --- chapter__aggregate
       R --- chapter__validation
       R --- chapter__conditional
       R --- chapter__clause
       R --- chapter__definitions
       click chapter__general href "general/index.html"
       click chapter__join href "join/index.html"
       click chapter__string href "string/index.html"
       click chapter__numeric href "numeric/index.html"
       click chapter__comparison href "comparison/index.html"
       click chapter__boolean href "boolean/index.html"
       click chapter__time href "time/index.html"
       click chapter__set href "set/index.html"
       click chapter__hierarchy href "hierarchy/index.html"
       click chapter__aggregate href "aggregate/index.html"
       click chapter__validation href "validation/index.html"
       click chapter__conditional href "conditional/index.html"
       click chapter__clause href "clause/index.html"
       click chapter__definitions href "definitions/index.html"

.. raw:: html

   </div>

.. rubric:: Operators

The blocks of operators, one for each chapter of operators of the manual. The counts are over the forms of the operators of each block. Their pages are under :doc:`Operators <operators>` in the sidebar.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Block
     - Status
     - Operators
     - Forms
     - 🟢
     - 🟡
     - 🛡️
     - 🔴
     - 🔵
   * - :doc:`General purpose operators <general/index>`
     - 🟡
     - 7
     - 62
     - 61
     - 1
     - 0
     - 0
     - 0
   * - :doc:`Join operators <join/index>`
     - 🟢
     - 4
     - 89
     - 89
     - 0
     - 0
     - 0
     - 0
   * - :doc:`String operators <string/index>`
     - 🟢
     - 11
     - 58
     - 58
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Numeric operators <numeric/index>`
     - 🟢
     - 18
     - 94
     - 94
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Comparison operators <comparison/index>`
     - 🟢
     - 12
     - 67
     - 67
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Boolean operators <boolean/index>`
     - 🟢
     - 4
     - 24
     - 24
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Time operators <time/index>`
     - 🟢
     - 17
     - 54
     - 54
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Set operators <set/index>`
     - 🟢
     - 4
     - 6
     - 6
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Hierarchical aggregation <hierarchy/index>`
     - 🟢
     - 1
     - 38
     - 38
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Aggregate and Analytic operators <aggregate/index>`
     - 🟢
     - 18
     - 112
     - 112
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Data validation operators <validation/index>`
     - 🟢
     - 3
     - 53
     - 53
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Conditional operators <conditional/index>`
     - 🟢
     - 3
     - 19
     - 19
     - 0
     - 0
     - 0
     - 0
   * - :doc:`Clause operators <clause/index>`
     - 🟢
     - 9
     - 24
     - 24
     - 0
     - 0
     - 0
     - 0
   * - **Total**
     - 🟡
     - 111
     - 700
     - 699
     - 1
     - 0
     - 0
     - 0

.. rubric:: Definitions (VTL-DL)

The definitions of the VTL-DL. The counts are over the forms of each one. Their pages are under :doc:`Definitions (VTL-DL) <definitions/index>` in the sidebar.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Definition
     - Status
     - Forms
     - 🟢
     - 🟡
     - 🛡️
     - 🔴
     - 🔵
   * - :doc:`define operator <definitions/define_operator>`
     - 🟢
     - 13
     - 13
     - 0
     - 0
     - 0
     - 0
   * - :doc:`define datapoint ruleset <definitions/define_datapoint_ruleset>`
     - 🟢
     - 5
     - 5
     - 0
     - 0
     - 0
     - 0
   * - :doc:`define hierarchical ruleset <definitions/define_hierarchical_ruleset>`
     - 🟢
     - 7
     - 7
     - 0
     - 0
     - 0
     - 0
   * - :doc:`define viral propagation <definitions/define_viral_propagation>`
     - 🟢
     - 4
     - 4
     - 0
     - 0
     - 0
     - 0
   * - **Total**
     - 🟢
     - 29
     - 29
     - 0
     - 0
     - 0
     - 0

.. rubric:: Not implemented

The codes of the errors of vtlengine that say that it does not implement something. A form whose examples fail with one of them on both engines is 🔴, not implemented, rather than 🛡️, refused. The list is kept by hand, in ``docs/operator_map/tool/status.py``, with the messages as the error catalogue of vtlengine writes them.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Code
     - Message
   * - ``1-3-13``
     - Unsupported VTL construct: {construct}.
   * - ``0-5-1-10``
     - Invalid parameter type definition {definition}.

.. rubric:: Attribution

Examples marked *manual* are taken from the VTL 2.2 Reference Manual (sdmx-twg/vtl @ 1bea1153beda0dc553767e7ca8fe16a0d51c5b34), converted to the vtlengine format.

The manual is published at `sdmx-twg.github.io/vtl/v2.2 <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/index.html>`__, and its source is the repository `sdmx-twg/vtl <https://github.com/sdmx-twg/vtl>`__.

.. toctree::
   :maxdepth: 1
   :hidden:

   operators
   definitions/index
