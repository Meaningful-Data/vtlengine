================================
String operators
================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 10 of 11 operators all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (27ad1da7) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["String operators"]:::block
       string__concat["String concatenation · #124;#124;<br/>🟢7"]:::ok
       string__trim["Whitespace removal · trim<br/>🟢3"]:::ok
       string__ltrim["Leading whitespace removal · ltrim<br/>🟢3"]:::ok
       string__rtrim["Trailing whitespace removal · rtrim<br/>🟢3"]:::ok
       string__upper["Upper case conversion · upper<br/>🟢3"]:::ok
       string__lower["Lower case conversion · lower<br/>🟢3"]:::ok
       string__substr["Sub-string extraction · substr<br/>🟢8"]:::ok
       string__replace["String pattern replacement · replace<br/>🟢6"]:::ok
       string__instr["String pattern location · instr<br/>🟢8"]:::ok
       string__length["String length · length<br/>🟢3"]:::ok
       string__string_distance["String distances · string_distance<br/>🟢10 🟡1"]:::partial
       string__concat --- B
       string__trim --- B
       string__ltrim --- B
       string__rtrim --- B
       string__upper --- B
       string__lower --- B
       B --- string__substr
       B --- string__replace
       B --- string__instr
       B --- string__length
       B --- string__string_distance
       click string__concat href "concat.html"
       click string__trim href "trim.html"
       click string__ltrim href "ltrim.html"
       click string__rtrim href "rtrim.html"
       click string__upper href "upper.html"
       click string__lower href "lower.html"
       click string__substr href "substr.html"
       click string__replace href "replace.html"
       click string__instr href "instr.html"
       click string__length href "length.html"
       click string__string_distance href "string_distance.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`String concatenation <concat>` ``||``
     - 🟢
     - 7
   * - :doc:`Whitespace removal <trim>` ``trim``
     - 🟢
     - 3
   * - :doc:`Leading whitespace removal <ltrim>` ``ltrim``
     - 🟢
     - 3
   * - :doc:`Trailing whitespace removal <rtrim>` ``rtrim``
     - 🟢
     - 3
   * - :doc:`Upper case conversion <upper>` ``upper``
     - 🟢
     - 3
   * - :doc:`Lower case conversion <lower>` ``lower``
     - 🟢
     - 3
   * - :doc:`Sub-string extraction <substr>` ``substr``
     - 🟢
     - 8
   * - :doc:`String pattern replacement <replace>` ``replace``
     - 🟢
     - 6
   * - :doc:`String pattern location <instr>` ``instr``
     - 🟢
     - 8
   * - :doc:`String length <length>` ``length``
     - 🟢
     - 3
   * - :doc:`String distances <string_distance>` ``string_distance``
     - 🟡
     - 11

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - integer
     - string
   * - :doc:`String concatenation <concat>`
     -
     -
     - in · out
   * - :doc:`Whitespace removal <trim>`
     -
     -
     - in · out
   * - :doc:`Leading whitespace removal <ltrim>`
     -
     -
     - in · out
   * - :doc:`Trailing whitespace removal <rtrim>`
     -
     -
     - in · out
   * - :doc:`Upper case conversion <upper>`
     -
     -
     - in · out
   * - :doc:`Lower case conversion <lower>`
     -
     -
     - in · out
   * - :doc:`Sub-string extraction <substr>`
     -
     - in
     - in · out
   * - :doc:`String pattern replacement <replace>`
     -
     -
     - in · out
   * - :doc:`String pattern location <instr>`
     -
     - in · out
     - in
   * - :doc:`String length <length>`
     -
     - out
     - in
   * - :doc:`String distances <string_distance>`
     - out
     -
     - in

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
   * - :doc:`String distances <string_distance>`
     - `string\_distance(jaro\_winkler, sc, sc) <string_distance.html#form-jaro-winkler-sc>`__
     - 🟡
     - `Identical strings <string_distance.html#ex-jaro-winkler-sc-identical-strings>`__
     - ❌
     - ❌
     - value expected 0 got 1.0

.. toctree::
   :maxdepth: 1
   :hidden:

   concat
   trim
   ltrim
   rtrim
   upper
   lower
   substr
   replace
   instr
   length
   string_distance
