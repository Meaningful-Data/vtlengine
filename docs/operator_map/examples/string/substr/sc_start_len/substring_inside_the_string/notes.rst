:Tests: ``substr`` of a scalar, with a ``start`` and a ``length`` that fit in the string.
:Manual: `Sub-string extraction <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Sub-string%20extraction/index.html>`__: "The substring starts from the startth character of the input string and has a number of characters equal to the ``length`` parameter.", as in ``substr ("abcdefghijklmnopqrstuvwxyz", 5 , 10 )`` gives ``"efghijklmn"``.
:Expected: ``SC_r`` is ``efghijklmn``, a String: the 10 characters from the 5th, ``e``, on.
