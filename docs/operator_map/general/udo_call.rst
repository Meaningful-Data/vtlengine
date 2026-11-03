================================================================================
User-defined operator call · ``name( )``
================================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 4 of 4 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (d5c65271) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["User-defined operator call · name( )<br/>🟢4"]:::ok
       v_no_args["op()"]:::ok
       v_all_args["op(a, b)"]:::ok
       v_default_omitted["op(a)"]:::ok
       v_placeholder_arg["op(a, _, c)"]:::ok
       v_no_args --- O
       v_all_args --- O
       O --- v_default_omitted
       O --- v_placeholder_arg
       click v_no_args href "#form-no-args"
       click v_all_args href "#form-all-args"
       click v_default_omitted href "#form-default-omitted"
       click v_placeholder_arg href "#form-placeholder-arg"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> operator, any scalar · <b>Gives</b> any type</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p><details class="om-dir"><summary><b>Comes from · operator</b> — 1 operator in 1 block 🟡1</summary><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a></div></details></details></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-no-args"></span>

op() 🟢 · operator → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-no-args-empty-argument-list"><summary><span class="om-ex-title">Empty argument list</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/udo_call/no_args/empty_argument_list/notes.rst

.. literalinclude:: ../examples/general/udo_call/no_args/empty_argument_list/script.vtl
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
     - ``"42"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/udo_call/no_args/empty_argument_list/script.vtl">script.vtl</a> · <a href="../examples/general/udo_call/no_args/empty_argument_list/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/udo_call/no_args/empty_argument_list/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-all-args"></span>

op(a, b) 🟢 · operator, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-all-args-positional-order"><summary><span class="om-ex-title">Positional order</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/udo_call/all_args/positional_order/notes.rst

.. literalinclude:: ../examples/general/udo_call/all_args/positional_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>10</td><td>3</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``"7"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/udo_call/all_args/positional_order/script.vtl">script.vtl</a> · <a href="../examples/general/udo_call/all_args/positional_order/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/general/udo_call/all_args/positional_order/input/scalars.json">input/scalars.json</a> · <a href="../examples/general/udo_call/all_args/positional_order/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/udo_call/all_args/positional_order/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-default-omitted"></span>

op(a) 🟢 · operator, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-default-omitted-trailing-default"><summary><span class="om-ex-title">Trailing default</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/udo_call/default_omitted/trailing_default/notes.rst

.. literalinclude:: ../examples/general/udo_call/default_omitted/trailing_default/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>10</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``"9"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/udo_call/default_omitted/trailing_default/script.vtl">script.vtl</a> · <a href="../examples/general/udo_call/default_omitted/trailing_default/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/general/udo_call/default_omitted/trailing_default/input/scalars.json">input/scalars.json</a> · <a href="../examples/general/udo_call/default_omitted/trailing_default/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/udo_call/default_omitted/trailing_default/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-placeholder-arg"></span>

op(a, \_, c) 🟢 · operator, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-placeholder-arg-underscore-in-the-middle"><summary><span class="om-ex-title">Underscore in the middle</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/udo_call/placeholder_arg/underscore_in_the_middle/notes.rst

.. literalinclude:: ../examples/general/udo_call/placeholder_arg/underscore_in_the_middle/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>10</td><td>5</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``"4"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/udo_call/placeholder_arg/underscore_in_the_middle/script.vtl">script.vtl</a> · <a href="../examples/general/udo_call/placeholder_arg/underscore_in_the_middle/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/general/udo_call/placeholder_arg/underscore_in_the_middle/input/scalars.json">input/scalars.json</a> · <a href="../examples/general/udo_call/placeholder_arg/underscore_in_the_middle/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/udo_call/placeholder_arg/underscore_in_the_middle/index.html">example page ↗</a></p>
   </details>
