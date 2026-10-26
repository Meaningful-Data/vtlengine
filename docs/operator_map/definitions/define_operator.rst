==============================
define operator
==============================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 6 of 13 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (9dc0a08a) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

.. raw:: html

   <div class="om-legend"><span class="om-chip ok">🟢 all ok</span><span class="om-chip partial">🟡 partially implemented</span><span class="om-chip refused">🛡️ refused with an error</span><span class="om-chip missing">🔴 not implemented or crash</span><span class="om-chip diverge">🔵 engines differ</span> <span class="om-key">✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash · 📌 team decision</span></div>

.. raw:: html

   <div class="om-diagram">

.. mermaid::
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart TB
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef refused fill:#d4edda,stroke:#28a745,stroke-width:3px,stroke-dasharray: 2 2,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef block fill:#2b3a4a,stroke:#2b3a4a,color:#fff,font-weight:bold
       classDef ext fill:#fff,stroke:#666,color:#000,stroke-dasharray: 5 5
       O["define operator<br/>🟢6 🟡2 🔴5"]:::partial
       subgraph G_b1["Signature"]
           direction LR
           v_no_parameters["f ()"]:::ok
           v_scalar_type["x #60;scalar type#62;"]:::ok
           v_value_domain["x #60;value domain#62;"]:::missing
           v_constrained_scalar["x #60;scalar type#62; #91;value …#93; #124; #123; … #125; #124; not null"]:::missing
           v_set["s set#60;…#62;"]:::missing
           v_component["c component #124; identifier #124; measure #124; attribute #124; viral attribute #60;…#62;"]:::partial
           v_dataset["ds dataset #91;#123; … #125;#93;"]:::partial
           v_ruleset["r #60;ruleset type#62;"]:::missing
           v_default_value["y #60;type#62; default …"]:::ok
       end
       subgraph G_b2["Return type"]
           direction LR
           v_returns_omitted["no returns"]:::missing
           v_returns_scalar["returns #60;scalar type#62;"]:::ok
           v_returns_component["returns component#60;…#62;"]:::ok
           v_returns_dataset["returns dataset"]:::ok
       end
       O --- G_b1
       O --- G_b2
       style G_b1 fill:#f8f9fa,stroke:#868e96,stroke-width:1px
       style G_b2 fill:#f8f9fa,stroke:#868e96,stroke-width:1px
       click v_no_parameters href "#form-no-parameters"
       click v_scalar_type href "#form-scalar-type"
       click v_value_domain href "#form-value-domain"
       click v_constrained_scalar href "#form-constrained-scalar"
       click v_set href "#form-set"
       click v_component href "#form-component"
       click v_dataset href "#form-dataset"
       click v_ruleset href "#form-ruleset"
       click v_default_value href "#form-default-value"
       click v_returns_omitted href "#form-returns-omitted"
       click v_returns_scalar href "#form-returns-scalar"
       click v_returns_component href "#form-returns-component"
       click v_returns_dataset href "#form-returns-dataset"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> number, integer, string, boolean, time, date, time_period, duration, set, value domain, datapoint ruleset, hierarchical ruleset, any scalar · <b>Gives</b> operator</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p><details class="om-dir" open><summary><b>Comes from · set, value domain, datapoint ruleset, hierarchical ruleset</b> — 2 operators in 1 block 🟡2</summary><details class="om-blk"><summary>Definitions (VTL-DL) <b>(2)</b> 🟡2</summary><div><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a></div></details></details></div><div class="om-col"><details class="om-dir" open><summary><b>Goes to · operator</b> — 1 operator in 1 block 🟢1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <p class="om-box-title">Signature</p>

.. raw:: html

   <span class="om-anchor" id="form-no-parameters"></span>

f () 🟢 · nothing → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-no-parameters-call-without-arguments" open><summary><span class="om-ex-title">Call without arguments</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/no_parameters/call_without_arguments/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/no_parameters/call_without_arguments/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``42``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/no_parameters/call_without_arguments/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/no_parameters/call_without_arguments/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/no_parameters/call_without_arguments/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-scalar-type"></span>

x <scalar type> 🟢 · any scalar, number, integer, string, boolean, time, date, time_period, duration → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-scalar-type-integer-argument" open><summary><span class="om-ex-title">Integer argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/integer_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/integer_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``3``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/integer_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/integer_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/integer_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-scalar-type-number-argument"><summary><span class="om-ex-title">Number argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/number_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/number_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Number
     - ``2.5``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/number_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/number_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/number_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-scalar-type-string-argument"><summary><span class="om-ex-title">String argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/string_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/string_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - String
     - ``"A_B"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/string_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/string_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/string_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-scalar-type-boolean-argument"><summary><span class="om-ex-title">Boolean argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/boolean_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/boolean_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Boolean
     - ``false``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/boolean_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/boolean_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/boolean_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-scalar-type-date-argument"><summary><span class="om-ex-title">Date argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/date_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/date_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Date
     - ``"2020-01-15"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/date_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/date_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/date_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-scalar-type-time-period-argument"><summary><span class="om-ex-title">Time period argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/time_period_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/time_period_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Time\_Period
     - ``"2020Q1"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/time_period_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/time_period_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/time_period_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-scalar-type-time-argument"><summary><span class="om-ex-title">Time argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/time_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/time_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Time
     - ``"2020-01-01/2020-12-31"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/time_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/time_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/time_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-scalar-type-duration-argument"><summary><span class="om-ex-title">Duration argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/duration_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/duration_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Duration
     - ``"M"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/duration_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/duration_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/duration_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-scalar-type-scalar-argument-keeps-its-type"><summary><span class="om-ex-title">Scalar argument keeps its type</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/scalar_type/scalar_argument_keeps_its_type/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/scalar_type/scalar_argument_keeps_its_type/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``5``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/scalar_type/scalar_argument_keeps_its_type/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/scalar_type/scalar_argument_keeps_its_type/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/scalar_type/scalar_argument_keeps_its_type/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-value-domain"></span>

x <value domain> 🔴 · value domain → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-value-domain-string-value-of-the-domain" open><summary><span class="om-ex-title">String value of the domain</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-10</code> · DuckDB 🛡 <code>0-5-1-10</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-10: Invalid parameter type definition &#x27;VD_1&#x27; (line 1, column 32).</p>

.. include:: ../examples/definitions/define_operator/value_domain/string_value_of_the_domain/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/value_domain/string_value_of_the_domain/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - String
     - ``"A_X"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/value_domain/string_value_of_the_domain/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/value_domain/string_value_of_the_domain/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/value_domain/string_value_of_the_domain/value_domains/VD_1.json">value_domains/VD_1.json</a> · <a href="../examples/definitions/define_operator/value_domain/string_value_of_the_domain/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-constrained-scalar"></span>

x <scalar type> [value …] \| { … } \| not null 🔴 · integer → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-constrained-scalar-value-condition-holds" open><summary><span class="om-ex-title">Value condition holds</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: type constraint in &#x27;integer [value &gt;= 0]&#x27; (line 1, column 30).</p>

.. include:: ../examples/definitions/define_operator/constrained_scalar/value_condition_holds/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/constrained_scalar/value_condition_holds/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``20``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/constrained_scalar/value_condition_holds/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/constrained_scalar/value_condition_holds/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/constrained_scalar/value_condition_holds/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-constrained-scalar-value-in-the-list"><summary><span class="om-ex-title">Value in the list</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: type constraint in &#x27;integer { 1, 2, 3 }&#x27; (line 1, column 30).</p>

.. include:: ../examples/definitions/define_operator/constrained_scalar/value_in_the_list/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/constrained_scalar/value_in_the_list/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``20``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/constrained_scalar/value_in_the_list/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/constrained_scalar/value_in_the_list/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/constrained_scalar/value_in_the_list/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-constrained-scalar-non-null-argument"><summary><span class="om-ex-title">Non null argument</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: not null in &#x27;integer not null&#x27; (line 1, column 28).</p>

.. include:: ../examples/definitions/define_operator/constrained_scalar/non_null_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/constrained_scalar/non_null_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``3``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/constrained_scalar/non_null_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/constrained_scalar/non_null_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/constrained_scalar/non_null_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-set"></span>

s set<…> 🔴 · set → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-set-value-domain-as-the-set" open><summary><span class="om-ex-title">Value domain as the set</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: set type &#x27;set&lt;string&gt;&#x27; (line 1, column 36).</p>

.. include:: ../examples/definitions/define_operator/set/value_domain_as_the_set/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/set/value_domain_as_the_set/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Boolean
     - ``true``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/set/value_domain_as_the_set/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/set/value_domain_as_the_set/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/set/value_domain_as_the_set/value_domains/VD_AB.json">value_domains/VD_AB.json</a> · <a href="../examples/definitions/define_operator/set/value_domain_as_the_set/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-component"></span>

c component \| identifier \| measure \| attribute \| viral attribute <…> 🟡 · any scalar, integer, string → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-component-component-role-in-calc" open><summary><span class="om-ex-title">Component role in calc</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/component/component_role_in_calc/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/component/component_role_in_calc/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>20</td></tr><tr><td>2</td><td>20</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/component/component_role_in_calc/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/component/component_role_in_calc/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/component/component_role_in_calc/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/component/component_role_in_calc/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/component/component_role_in_calc/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/component/component_role_in_calc/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-component-identifier-in-calc"><summary><span class="om-ex-title">Identifier in calc</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/component/identifier_in_calc/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/component/identifier_in_calc/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>10</td></tr><tr><td>2</td><td>20</td><td>20</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/component/identifier_in_calc/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/component/identifier_in_calc/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/component/identifier_in_calc/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/component/identifier_in_calc/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/component/identifier_in_calc/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/component/identifier_in_calc/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-component-measure-role-in-calc"><summary><span class="om-ex-title">Measure role in calc</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/component/measure_role_in_calc/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/component/measure_role_in_calc/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>20</td></tr><tr><td>2</td><td>20</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/component/measure_role_in_calc/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/component/measure_role_in_calc/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/component/measure_role_in_calc/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/component/measure_role_in_calc/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/component/measure_role_in_calc/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/component/measure_role_in_calc/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-component-attribute-overwritten-in-calc"><summary><span class="om-ex-title">Attribute overwritten in calc</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">role of At_1 expected Attribute got Measure</p>

.. include:: ../examples/definitions/define_operator/component/attribute_overwritten_in_calc/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/component/attribute_overwritten_in_calc/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>A</td></tr><tr><td>2</td><td>20</td><td>B</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>A_X</td></tr><tr><td>2</td><td>20</td><td>B_X</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/component/attribute_overwritten_in_calc/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/component/attribute_overwritten_in_calc/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/component/attribute_overwritten_in_calc/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/component/attribute_overwritten_in_calc/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/component/attribute_overwritten_in_calc/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/component/attribute_overwritten_in_calc/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-component-viral-attribute-overwritten-in-calc"><summary><span class="om-ex-title">Viral attribute overwritten in calc</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">role of At_1 expected Viral Attribute got Measure</p>

.. include:: ../examples/definitions/define_operator/component/viral_attribute_overwritten_in_calc/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/component/viral_attribute_overwritten_in_calc/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>A</td></tr><tr><td>2</td><td>20</td><td>B</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>A_X</td></tr><tr><td>2</td><td>20</td><td>B_X</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/component/viral_attribute_overwritten_in_calc/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/component/viral_attribute_overwritten_in_calc/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/component/viral_attribute_overwritten_in_calc/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/component/viral_attribute_overwritten_in_calc/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/component/viral_attribute_overwritten_in_calc/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/component/viral_attribute_overwritten_in_calc/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-dataset"></span>

ds dataset [{ … }] 🟡 · any scalar → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-dataset-dataset-argument" open><summary><span class="om-ex-title">Dataset argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/dataset/dataset_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/dataset/dataset_argument/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>20</td></tr><tr><td>2</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/dataset/dataset_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/dataset/dataset_argument/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/dataset/dataset_argument/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/dataset/dataset_argument/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/dataset/dataset_argument/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/dataset/dataset_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-dataset-named-components"><summary><span class="om-ex-title">Named components</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/dataset/named_components/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/dataset/named_components/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>20</td></tr><tr><td>2</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/dataset/named_components/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/dataset/named_components/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/dataset/named_components/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/dataset/named_components/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/dataset/named_components/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/dataset/named_components/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-dataset-exactly-one-measure"><summary><span class="om-ex-title">Exactly one measure</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: multiplicity modifier in &#x27;measure&lt;integer&gt; _&#x27; (line 1, column 37).</p>

.. include:: ../examples/definitions/define_operator/dataset/exactly_one_measure/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/dataset/exactly_one_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>20</td></tr><tr><td>2</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/dataset/exactly_one_measure/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/dataset/exactly_one_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/dataset/exactly_one_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/dataset/exactly_one_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/dataset/exactly_one_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/dataset/exactly_one_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-dataset-one-or-more-measures"><summary><span class="om-ex-title">One or more measures</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: multiplicity modifier in &#x27;measure&lt;integer&gt; _+&#x27; (line 1, column 37).</p>

.. include:: ../examples/definitions/define_operator/dataset/one_or_more_measures/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/dataset/one_or_more_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>1</td></tr><tr><td>2</td><td>20</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>20</td><td>2</td></tr><tr><td>2</td><td>40</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/dataset/one_or_more_measures/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/dataset/one_or_more_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/dataset/one_or_more_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/dataset/one_or_more_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/dataset/one_or_more_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/dataset/one_or_more_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-dataset-zero-or-more-measures"><summary><span class="om-ex-title">Zero or more measures</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: multiplicity modifier in &#x27;measure&lt;integer&gt; _*&#x27; (line 1, column 37).</p>

.. include:: ../examples/definitions/define_operator/dataset/zero_or_more_measures/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/dataset/zero_or_more_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>20</td></tr><tr><td>2</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/dataset/zero_or_more_measures/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/dataset/zero_or_more_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/dataset/zero_or_more_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/dataset/zero_or_more_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/dataset/zero_or_more_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/dataset/zero_or_more_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ruleset"></span>

r <ruleset type> 🔴 · datapoint ruleset, hierarchical ruleset → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ruleset-datapoint-type" open><summary><span class="om-ex-title">Datapoint type</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: ruleset type &#x27;datapoint&#x27; (line 5, column 49).</p>

.. include:: ../examples/definitions/define_operator/ruleset/datapoint_type/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/ruleset/datapoint_type/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>-5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>true</td><td></td><td></td></tr><tr><td>2</td><td>1</td><td>false</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/ruleset/datapoint_type/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_type/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_type/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_type/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_type/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_type/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ruleset-datapoint-on-variables-type"><summary><span class="om-ex-title">Datapoint on variables type</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: ruleset type &#x27;datapoint_on_variables&#x27; (line 5, column 49).</p>

.. include:: ../examples/definitions/define_operator/ruleset/datapoint_on_variables_type/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/ruleset/datapoint_on_variables_type/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>-5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>true</td><td></td><td></td></tr><tr><td>2</td><td>1</td><td>false</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/ruleset/datapoint_on_variables_type/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_variables_type/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_variables_type/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_variables_type/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_variables_type/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_variables_type/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ruleset-datapoint-on-valuedomains-type"><summary><span class="om-ex-title">Datapoint on valuedomains type</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: ruleset type &#x27;datapoint_on_valuedomains&#x27; (line 5, column 49).</p>

.. include:: ../examples/definitions/define_operator/ruleset/datapoint_on_valuedomains_type/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/ruleset/datapoint_on_valuedomains_type/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>-5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>true</td><td></td><td></td></tr><tr><td>2</td><td>1</td><td>false</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/ruleset/datapoint_on_valuedomains_type/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_valuedomains_type/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_valuedomains_type/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_valuedomains_type/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_valuedomains_type/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/ruleset/datapoint_on_valuedomains_type/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ruleset-generic-ruleset-type"><summary><span class="om-ex-title">Generic ruleset type</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: ruleset type &#x27;ruleset&#x27; (line 5, column 49).</p>

.. include:: ../examples/definitions/define_operator/ruleset/generic_ruleset_type/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/ruleset/generic_ruleset_type/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>-5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>true</td><td></td><td></td></tr><tr><td>2</td><td>1</td><td>false</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/ruleset/generic_ruleset_type/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/ruleset/generic_ruleset_type/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/ruleset/generic_ruleset_type/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/ruleset/generic_ruleset_type/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/ruleset/generic_ruleset_type/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/ruleset/generic_ruleset_type/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ruleset-hierarchical-type"><summary><span class="om-ex-title">Hierarchical type</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: ruleset type &#x27;hierarchical&#x27; (line 5, column 45).</p>

.. include:: ../examples/definitions/define_operator/ruleset/hierarchical_type/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/ruleset/hierarchical_type/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>4</td></tr><tr><td>1</td><td>C</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/ruleset/hierarchical_type/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_type/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_type/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_type/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_type/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_type/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ruleset-hierarchical-on-variables-type"><summary><span class="om-ex-title">Hierarchical on variables type</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: ruleset type &#x27;hierarchical_on_variables&#x27; (line 5, column 45).</p>

.. include:: ../examples/definitions/define_operator/ruleset/hierarchical_on_variables_type/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/ruleset/hierarchical_on_variables_type/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>4</td></tr><tr><td>1</td><td>C</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_variables_type/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_variables_type/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_variables_type/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_variables_type/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_variables_type/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_variables_type/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ruleset-hierarchical-on-valuedomains-type"><summary><span class="om-ex-title">Hierarchical on valuedomains type</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: ruleset type &#x27;hierarchical_on_valuedomains&#x27; (line 5, column 45).</p>

.. include:: ../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>4</td></tr><tr><td>1</td><td>C</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/value_domains/VD_ITEM.json">value_domains/VD_ITEM.json</a> · <a href="../examples/definitions/define_operator/ruleset/hierarchical_on_valuedomains_type/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-default-value"></span>

y <type> default … 🟢 · integer → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-default-value-last-argument-omitted" open><summary><span class="om-ex-title">Last argument omitted</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/default_value/last_argument_omitted/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/default_value/last_argument_omitted/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``11``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/default_value/last_argument_omitted/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/default_value/last_argument_omitted/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/default_value/last_argument_omitted/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-default-value-underscore-for-the-argument"><summary><span class="om-ex-title">Underscore for the argument</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/default_value/underscore_for_the_argument/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/default_value/underscore_for_the_argument/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``11``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/default_value/underscore_for_the_argument/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/default_value/underscore_for_the_argument/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/default_value/underscore_for_the_argument/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <p class="om-box-title">Return type</p>

.. raw:: html

   <span class="om-anchor" id="form-returns-omitted"></span>

no returns 🔴 · any scalar → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-returns-omitted-type-of-the-body" open><summary><span class="om-ex-title">Type of the body</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-2-2</code> · DuckDB 🛡 <code>1-3-2-2</code></span></summary>
   <p class="om-detail">SemanticError 1-3-2-2: At op add_one: User defined operator without returns is not implemented.</p>

.. include:: ../examples/definitions/define_operator/returns_omitted/type_of_the_body/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/returns_omitted/type_of_the_body/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``3``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/returns_omitted/type_of_the_body/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/returns_omitted/type_of_the_body/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/returns_omitted/type_of_the_body/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-returns-scalar"></span>

returns <scalar type> 🟢 · integer → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-returns-scalar-larger-of-two-integers" open><summary><span class="om-ex-title">Larger of two integers</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/returns_scalar/larger_of_two_integers/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/returns_scalar/larger_of_two_integers/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``3``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/returns_scalar/larger_of_two_integers/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/returns_scalar/larger_of_two_integers/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/definitions/define_operator/returns_scalar/larger_of_two_integers/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-returns-component"></span>

returns component<…> 🟢 · integer → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-returns-component-component-in-calc" open><summary><span class="om-ex-title">Component in calc</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/returns_component/component_in_calc/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/returns_component/component_in_calc/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>11</td></tr><tr><td>2</td><td>20</td><td>21</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/returns_component/component_in_calc/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/returns_component/component_in_calc/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/returns_component/component_in_calc/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/returns_component/component_in_calc/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/returns_component/component_in_calc/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/returns_component/component_in_calc/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-returns-dataset"></span>

returns dataset 🟢 · any scalar → operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-returns-dataset-every-measure-plus-one" open><summary><span class="om-ex-title">Every measure plus one</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_operator/returns_dataset/every_measure_plus_one/notes.rst

.. literalinclude:: ../examples/definitions/define_operator/returns_dataset/every_measure_plus_one/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>11</td></tr><tr><td>2</td><td>21</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_operator/returns_dataset/every_measure_plus_one/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_operator/returns_dataset/every_measure_plus_one/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_operator/returns_dataset/every_measure_plus_one/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_operator/returns_dataset/every_measure_plus_one/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_operator/returns_dataset/every_measure_plus_one/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_operator/returns_dataset/every_measure_plus_one/index.html">example page ↗</a></p>
   </details>
