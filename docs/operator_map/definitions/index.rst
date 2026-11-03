========================================
Definitions (VTL-DL)
========================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 0 of 4 operators all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (d5c65271) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["Definitions (VTL-DL)"]:::block
       definitions__define_operator["define operator<br/>🟢9 🟡1 🔴3"]:::partial
       definitions__define_datapoint_ruleset["define datapoint ruleset<br/>🟢3 🟡1 🛡️1"]:::partial
       definitions__define_hierarchical_ruleset["define hierarchical ruleset<br/>🟢6 🟡1"]:::partial
       definitions__define_viral_propagation["define viral propagation<br/>🟢3 🔴1"]:::partial
       definitions__define_operator --- B
       definitions__define_datapoint_ruleset --- B
       B --- definitions__define_hierarchical_ruleset
       B --- definitions__define_viral_propagation
       click definitions__define_operator href "define_operator.html"
       click definitions__define_datapoint_ruleset href "define_datapoint_ruleset.html"
       click definitions__define_hierarchical_ruleset href "define_hierarchical_ruleset.html"
       click definitions__define_viral_propagation href "define_viral_propagation.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`define operator <define_operator>`
     - 🟡
     - 13
   * - :doc:`define datapoint ruleset <define_datapoint_ruleset>`
     - 🟡
     - 5
   * - :doc:`define hierarchical ruleset <define_hierarchical_ruleset>`
     - 🟡
     - 7
   * - :doc:`define viral propagation <define_viral_propagation>`
     - 🟡
     - 4

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - integer
     - string
     - boolean
     - time
     - date
     - time_period
     - duration
     - set
     - value domain
     - datapoint ruleset
     - hierarchical ruleset
     - viral propagation
     - operator
     - any scalar
   * - :doc:`define operator <define_operator>`
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     - in
     -
     - out
     - in
   * - :doc:`define datapoint ruleset <define_datapoint_ruleset>`
     -
     - in
     - in
     - in
     -
     -
     -
     -
     -
     - in
     - out
     -
     -
     -
     -
   * - :doc:`define hierarchical ruleset <define_hierarchical_ruleset>`
     -
     - in
     - in
     - in
     -
     -
     -
     -
     -
     - in
     -
     - out
     -
     -
     -
   * - :doc:`define viral propagation <define_viral_propagation>`
     - in
     -
     -
     -
     -
     -
     -
     -
     -
     - in
     -
     -
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
   * - :doc:`define operator <define_operator>`
     - `x <value domain> <define_operator.html#form-value-domain>`__
     - 🔴
     - `String value of the domain <define_operator.html#ex-value-domain-string-value-of-the-domain>`__
     - 🛡 0-5-1-10
     - 🛡 0-5-1-10
     - InputValidationException 0-5-1-10: Invalid parameter type definition \'VD\_1\' (line 1, column 32).
   * - :doc:`define operator <define_operator>`
     - `s set<…> <define_operator.html#form-set>`__
     - 🔴
     - `Value domain as the set <define_operator.html#ex-set-value-domain-as-the-set>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: set type \'set<string>\' (line 1, column 36).
   * - :doc:`define operator <define_operator>`
     - `c component \| identifier \| measure \| attribute \| viral attribute <…> <define_operator.html#form-component>`__
     - 🟡
     - `Attribute overwritten in calc <define_operator.html#ex-component-attribute-overwritten-in-calc>`__
     - ❌
     - ❌
     - role of At\_1 expected Attribute got Measure
   * - :doc:`define operator <define_operator>`
     - `r <ruleset type> <define_operator.html#form-ruleset>`__
     - 🔴
     - `Datapoint type <define_operator.html#ex-ruleset-datapoint-type>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: ruleset type \'datapoint\' (line 5, column 49).
   * - :doc:`define datapoint ruleset <define_datapoint_ruleset>`
     - `(valuedomain VD [as alias], …) <define_datapoint_ruleset.html#form-valuedomain>`__
     - 🛡️
     - `Domains bound to components <define_datapoint_ruleset.html#ex-valuedomain-domains-bound-to-components>`__
     - 🛡 1-1-1-10
     - 🛡 1-1-1-10
     - SemanticError 1-1-1-10: Component VD\_FLOW not found in Dataset DS\_1. Please check transformation with output Dataset DS\_r
   * - :doc:`define datapoint ruleset <define_datapoint_ruleset>`
     - `[name :] … [errorcode …] [errorlevel …] <define_datapoint_ruleset.html#form-rule-options>`__
     - 🟡
     - `Errorlevel on the invalid point <define_datapoint_ruleset.html#ex-rule-options-errorlevel-on-the-invalid-point>`__
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
   * - :doc:`define hierarchical ruleset <define_hierarchical_ruleset>`
     - `[name :] … [errorcode …] [errorlevel …] <define_hierarchical_ruleset.html#form-rule-options>`__
     - 🟡
     - `Errorlevel on the failed rule <define_hierarchical_ruleset.html#ex-rule-options-errorlevel-on-the-failed-rule>`__
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
   * - :doc:`define viral propagation <define_viral_propagation>`
     - `(valuedomain VD) <define_viral_propagation.html#form-valuedomain>`__
     - 🔴
     - `Inherited from the domain <define_viral_propagation.html#ex-valuedomain-inherited-from-the-domain>`__
     - 🛡 1-3-3-8
     - 🛡 1-3-3-8
     - SemanticError 1-3-3-8: Viral propagation definition VP\_1 is defined on value domain CL\_CONF, which is not supported yet; define it on a variable instead.

.. toctree::
   :maxdepth: 1
   :hidden:

   define_operator
   define_datapoint_ruleset
   define_hierarchical_ruleset
   define_viral_propagation
