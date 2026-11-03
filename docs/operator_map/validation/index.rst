==================================================
Data validation operators
==================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 1 of 3 operators all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (55780655) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["Data validation operators"]:::block
       validation__check_datapoint["Check datapoint · check_datapoint<br/>🟢3 🟡4"]:::partial
       validation__check_hierarchy["Check hierarchy · check_hierarchy<br/>🟢28 🟡12"]:::partial
       validation__check["Check · check<br/>🟢6"]:::ok
       validation__check_datapoint --- B
       validation__check_hierarchy --- B
       B --- validation__check
       click validation__check_datapoint href "check_datapoint.html"
       click validation__check_hierarchy href "check_hierarchy.html"
       click validation__check href "check.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Check datapoint <check_datapoint>` ``check_datapoint``
     - 🟡
     - 7
   * - :doc:`Check hierarchy <check_hierarchy>` ``check_hierarchy``
     - 🟡
     - 40
   * - :doc:`Check <check>` ``check``
     - 🟢
     - 6

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - integer
     - string
     - boolean
     - datapoint ruleset
     - hierarchical ruleset
     - any scalar
     - any type
   * - :doc:`Check datapoint <check_datapoint>`
     -
     - out
     - out
     - out
     - in
     -
     - in
     - out
   * - :doc:`Check hierarchy <check_hierarchy>`
     - in · out
     - out
     - out
     - out
     -
     - in
     -
     -
   * - :doc:`Check <check>`
     - in · out
     - in · out
     - in · out
     - in · out
     -
     -
     -
     -

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
   * - :doc:`Check datapoint <check_datapoint>`
     - `check\_datapoint(DS, DPR) <check_datapoint.html#form-output-default>`__
     - 🟡
     - `Manual example 1 <check_datapoint.html#ex-output-default-manual-ex-1>`__
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
   * - :doc:`Check datapoint <check_datapoint>`
     - `check\_datapoint(DS, DPR all) <check_datapoint.html#form-all>`__
     - 🟡
     - `Manual example 2 <check_datapoint.html#ex-all-manual-ex-2>`__
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
   * - :doc:`Check datapoint <check_datapoint>`
     - `check\_datapoint(DS, DPR components C1, C2) <check_datapoint.html#form-components>`__
     - 🟡
     - `Value domains bound to components <check_datapoint.html#ex-components-value-domains-bound-to-components>`__
     - 🛡 1-1-1-10
     - 🛡 1-1-1-10
     - SemanticError 1-1-1-10: Component VD\_2 not found in Dataset DS\_1. Please check transformation with output Dataset DS\_r
   * - :doc:`Check datapoint <check_datapoint>`
     - `check\_datapoint(DS, DPR all), rules with errorlevel <check_datapoint.html#form-errorlevel>`__
     - 🟡
     - `Integer level on false results <check_datapoint.html#ex-errorlevel-integer-level-on-false-results>`__
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_null dataset all) <check_hierarchy.html#form-always-null-dataset-all>`__
     - 🟡
     - `Rule with no item found is null <check_hierarchy.html#ex-always-null-dataset-all-rule-with-no-item-found-is-null>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_null dataset all\_measures) <check_hierarchy.html#form-always-null-dataset-all-measures>`__
     - 🟡
     - `Rule with no item found is null <check_hierarchy.html#ex-always-null-dataset-all-measures-rule-with-no-item-found-is-null>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_null dataset\_priority all) <check_hierarchy.html#form-always-null-dataset-priority-all>`__
     - 🟡
     - `Null item from the rule computing it <check_hierarchy.html#ex-always-null-dataset-priority-all-null-item-from-the-rule-computing-it>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_null dataset\_priority all\_measures) <check_hierarchy.html#form-always-null-dataset-priority-all-measures>`__
     - 🟡
     - `Null item from the rule computing it <check_hierarchy.html#ex-always-null-dataset-priority-all-measures-null-item-from-the-rule-computing-it>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_zero dataset invalid) <check_hierarchy.html#form-always-zero-dataset-invalid>`__
     - 🟡
     - `Rule with no item found is false <check_hierarchy.html#ex-always-zero-dataset-invalid-rule-with-no-item-found-is-false>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_zero dataset all) <check_hierarchy.html#form-always-zero-dataset-all>`__
     - 🟡
     - `Rule with no item found is false <check_hierarchy.html#ex-always-zero-dataset-all-rule-with-no-item-found-is-false>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_zero dataset all\_measures) <check_hierarchy.html#form-always-zero-dataset-all-measures>`__
     - 🟡
     - `Rule with no item found is false <check_hierarchy.html#ex-always-zero-dataset-all-measures-rule-with-no-item-found-is-false>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_zero dataset\_priority invalid) <check_hierarchy.html#form-always-zero-dataset-priority-invalid>`__
     - 🟡
     - `Null item from the rule computing it <check_hierarchy.html#ex-always-zero-dataset-priority-invalid-null-item-from-the-rule-computing-it>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_zero dataset\_priority all) <check_hierarchy.html#form-always-zero-dataset-priority-all>`__
     - 🟡
     - `Null item from the rule computing it <check_hierarchy.html#ex-always-zero-dataset-priority-all-null-item-from-the-rule-computing-it>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR always\_zero dataset\_priority all\_measures) <check_hierarchy.html#form-always-zero-dataset-priority-all-measures>`__
     - 🟡
     - `Null item from the rule computing it <check_hierarchy.html#ex-always-zero-dataset-priority-all-measures-null-item-from-the-rule-computing-it>`__
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR rule C) <check_hierarchy.html#form-rule>`__
     - 🟡
     - `Manual example 1 <check_hierarchy.html#ex-rule-manual-ex-1>`__
     - 🛡 1-3-2-3
     - 🛡 1-3-2-3
     - SemanticError 1-3-2-3: At op hierarchy: Vtl Script contains Cycles, no DAG established. Nodes involved: {2: StatementDeps(inputs=[\'M\', \'N\', \'O\'], outputs=[\'B\'], persistent=[], unknown\_variables=[]), …
   * - :doc:`Check hierarchy <check_hierarchy>`
     - `check\_hierarchy(DS, HR partial\_zero dataset all), rules with errorlevel <check_hierarchy.html#form-errorlevel>`__
     - 🟡
     - `Integer level on false results <check_hierarchy.html#ex-errorlevel-integer-level-on-false-results>`__
     - ❌
     - ❌
     - type of imbalance expected Integer got Number; type of errorlevel expected Integer got Number

.. toctree::
   :maxdepth: 1
   :hidden:

   check_datapoint
   check_hierarchy
   check
