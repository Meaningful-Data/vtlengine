==================================================
General purpose operators
==================================================

.. raw:: html

   <div class="om-head"><span class="om-chip diverge">🔵 5 of 7 operators all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (ce4408d3) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["General purpose operators"]:::block
       general__parentheses["Parentheses · ( )<br/>🟢3"]:::ok
       general__persistent_assignment["Persistent assignment · #60;-<br/>🟢3"]:::ok
       general__non_persistent_assignment["Non-persistent assignment · #58;#61;<br/>🟢2"]:::ok
       general__membership["Membership · #35;<br/>🟢4 🟡1"]:::partial
       general__udo_call["User-defined operator call · name( )<br/>🟢4"]:::ok
       general__eval["Evaluation of an external routine · eval<br/>🟢4"]:::ok
       general__cast["Type conversion · cast<br/>🟢21 🟡2 🔴14 🔵4"]:::diverge
       general__parentheses --- B
       general__persistent_assignment --- B
       general__non_persistent_assignment --- B
       general__membership --- B
       B --- general__udo_call
       B --- general__eval
       B --- general__cast
       click general__parentheses href "parentheses.html"
       click general__persistent_assignment href "persistent_assignment.html"
       click general__non_persistent_assignment href "non_persistent_assignment.html"
       click general__membership href "membership.html"
       click general__udo_call href "udo_call.html"
       click general__eval href "eval.html"
       click general__cast href "cast.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Parentheses <parentheses>` ``( )``
     - 🟢
     - 3
   * - :doc:`Persistent assignment <persistent_assignment>` ``<-``
     - 🟢
     - 3
   * - :doc:`Non-persistent assignment <non_persistent_assignment>` ``:=``
     - 🟢
     - 2
   * - :doc:`Membership <membership>` ``#``
     - 🟡
     - 5
   * - :doc:`User-defined operator call <udo_call>` ``name( )``
     - 🟢
     - 4
   * - :doc:`Evaluation of an external routine <eval>` ``eval``
     - 🟢
     - 4
   * - :doc:`Type conversion <cast>` ``cast``
     - 🔵
     - 41

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
     - value domain
     - operator
     - external routine
     - any scalar
     - any type
   * - :doc:`Parentheses <parentheses>`
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     - in
     - out
   * - :doc:`Persistent assignment <persistent_assignment>`
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     - in
     - out
   * - :doc:`Non-persistent assignment <non_persistent_assignment>`
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     - in
     - out
   * - :doc:`Membership <membership>`
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     - in
     - out
   * - :doc:`User-defined operator call <udo_call>`
     -
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
     - in
     - out
   * - :doc:`Evaluation of an external routine <eval>`
     -
     - out
     - out
     -
     -
     -
     -
     -
     -
     -
     - in
     - in
     - out
   * - :doc:`Type conversion <cast>`
     - in · out
     - in · out
     - in · out
     - in · out
     - in · out
     - in · out
     - in · out
     - in · out
     - in
     -
     -
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
   * - :doc:`Membership <membership>`
     - `DS#At (viral) <membership.html#form-viral-attribute>`__
     - 🟡
     - `Manual example 6 <membership.html#ex-viral-attribute-manual-ex-6>`__
     - ❌
     - ❌
     - missing components: str\_var
   * - :doc:`Type conversion <cast>`
     - `cast(comp<boolean>, string) <cast.html#form-boolean-to-string>`__
     - 🟡
     - `True and false <cast.html#ex-boolean-to-string-true-and-false>`__
     - ❌
     - ❌
     - row Id\_1=1: Me\_2 expected true got True; row Id\_1=2: Me\_2 expected false got False
   * - :doc:`Type conversion <cast>`
     - `cast(comp<date>, time) <cast.html#form-date-to-time>`__
     - 🔵
     - `Same start and end <cast.html#ex-date-to-time-same-start-and-end>`__
     - ✅
     - ❌
     - DuckDB: row Id\_1=1: Me\_2 expected 2020-01-15/2020-01-15 got 2020-01-15 00:00:00; row Id\_1=2: Me\_2 expected 2021-12-31/2021-12-31 got 2021-12-31 00:00:00
   * - :doc:`Type conversion <cast>`
     - `cast(comp<time\_period>, time) <cast.html#form-time-period-to-time>`__
     - 🔵
     - `Start and end of the period <cast.html#ex-time-period-to-time-start-and-end-of-the-period>`__
     - ✅
     - ❌
     - DuckDB: row Id\_1=1: Me\_2 expected 2020-01-01/2020-03-31 got 2020-Q1; row Id\_1=2: Me\_2 expected 2021-02-01/2021-02-28 got 2021-M02
   * - :doc:`Type conversion <cast>`
     - `cast(comp<integer>, string, mask) <cast.html#form-integer-to-string-mask>`__
     - 🔴
     - `Two digit mask <cast.html#ex-integer-to-string-mask-two-digit-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"DD\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<number>, string, mask) <cast.html#form-number-to-string-mask>`__
     - 🔴
     - `Decimal mask <cast.html#ex-number-to-string-mask-decimal-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"DD.DDD\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<time>, string, mask) <cast.html#form-time-to-string-mask>`__
     - 🔴
     - `Start and end mask <cast.html#ex-time-to-string-mask-start-and-end-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY-MM-DD/YYYY-MM-DD\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<date>, string) <cast.html#form-date-to-string>`__
     - 🔵
     - `Iso 8601 dates <cast.html#ex-date-to-string-iso-8601-dates>`__
     - ✅
     - ❌
     - DuckDB: row Id\_1=1: Me\_2 expected 2020-01-15 got 2020-01-15 00:00:00; row Id\_1=2: Me\_2 expected 2021-12-31 got 2021-12-31 00:00:00
   * - :doc:`Type conversion <cast>`
     - `cast(comp<date>, string, mask) <cast.html#form-date-to-string-mask>`__
     - 🔴
     - `Year month day mask <cast.html#ex-date-to-string-mask-year-month-day-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYYMMDD\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<time\_period>, string) <cast.html#form-time-period-to-string>`__
     - 🟡
     - `Start and end dates <cast.html#ex-time-period-to-string-start-and-end-dates>`__
     - ❌
     - ❌
     - row Id\_1=1: Me\_2 expected 2020-01-01/2020-03-31 got 2020Q1; row Id\_1=2: Me\_2 expected 2021-02-01/2021-02-28 got 2021M2
   * - :doc:`Type conversion <cast>`
     - `cast(comp<time\_period>, string, mask) <cast.html#form-time-period-to-string-mask>`__
     - 🔴
     - `Monthly period mask <cast.html#ex-time-period-to-string-mask-monthly-period-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY\\MMM\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<duration>, string) <cast.html#form-duration-to-string>`__
     - 🔵
     - `Iso 8601 durations <cast.html#ex-duration-to-string-iso-8601-durations>`__
     - ✅
     - ❌
     - DuckDB: row Id\_1=1: Me\_2 expected P1Y got A; row Id\_1=2: Me\_2 expected P1M got M; row Id\_1=3: Me\_2 expected P1D got D
   * - :doc:`Type conversion <cast>`
     - `cast(comp<duration>, string, mask) <cast.html#form-duration-to-string-mask>`__
     - 🔴
     - `Period indicator mask <cast.html#ex-duration-to-string-mask-period-indicator-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"P\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<string>, integer, mask) <cast.html#form-string-to-integer-mask>`__
     - 🔴
     - `Two digit mask <cast.html#ex-string-to-integer-mask-two-digit-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"DD\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<string>, number, mask) <cast.html#form-string-to-number-mask>`__
     - 🔴
     - `Decimal mask <cast.html#ex-string-to-number-mask-decimal-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"DD.DDD\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<string>, time, mask) <cast.html#form-string-to-time-mask>`__
     - 🔴
     - `Start and end mask <cast.html#ex-string-to-time-mask-start-and-end-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY-MM-DD/YYYY-MM-DD\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<string>, date, mask) <cast.html#form-string-to-date-mask>`__
     - 🔴
     - `Year month day mask <cast.html#ex-string-to-date-mask-year-month-day-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY-MM-DD\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<string>, time\_period, mask) <cast.html#form-string-to-time-period-mask>`__
     - 🔴
     - `Quarter mask <cast.html#ex-string-to-time-period-mask-quarter-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY\\QQ\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<string>, duration, mask) <cast.html#form-string-to-duration-mask>`__
     - 🔴
     - `Period indicator mask <cast.html#ex-string-to-duration-mask-period-indicator-mask>`__
     - 🛡 1-3-13
     - 🛡 1-3-13
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"P\". Please check transformation with output Dataset DS\_r
   * - :doc:`Type conversion <cast>`
     - `cast(comp<string>, VD) <cast.html#form-string-to-value-domain>`__
     - 🔴
     - `Values in the domain <cast.html#ex-string-to-value-domain-values-in-the-domain>`__
     - 🛡 0-5-1-8
     - 🛡 0-5-1-8
     - InputValidationException 0-5-1-8: Value Domain \'VD\_1\' not available for cast operator or scalar type representation or rulesets.
   * - :doc:`Type conversion <cast>`
     - `cast(comp<VD>, VD) <cast.html#form-value-domain-to-value-domain>`__
     - 🔴
     - `Values in both domains <cast.html#ex-value-domain-to-value-domain-values-in-both-domains>`__
     - 🛡 0-5-1-8
     - 🛡 0-5-1-8
     - InputValidationException 0-5-1-8: Value Domain \'VD\_1\' not available for cast operator or scalar type representation or rulesets.

.. toctree::
   :maxdepth: 1
   :hidden:

   parentheses
   persistent_assignment
   non_persistent_assignment
   membership
   udo_call
   eval
   cast
