:Tests: an operator with no parameters, called as ``answer()``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``define operator operator_name ( { parameter { , parameter }* } )``, with optional parameters, so the call ``operatorName ( { argument { , argument }* } )`` passes none. The script ends with ``end operator``, as the page's examples and the grammar do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 42``, an Integer: ``6 * 7``.
