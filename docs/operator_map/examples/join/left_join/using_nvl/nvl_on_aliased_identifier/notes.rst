:Tests: ``using Id_1`` with ``nvl(d2#Id_2, "X")``, which names the ``Id_2`` of ``DS_2`` through its alias; the renames name the two ``Id_2``, as stripping their prefixes would give two ``Id_2``, an error.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: "prefixing the identifier name with the alias of the operand in the nvl() rule allows to specify different rules for each of them".
:Expected: the three data points of ``DS_1``: ``(1,A,C,10,1)``, ``(1,B,C,20,1)`` and ``(2,B,X,30,null)`` as ``Id_1``, ``Id_21``, ``Id_22``, ``Me_1``, ``Me_2``; ``Id_1`` = 3 is in ``DS_2`` only.
:Finding: both engines crash with AttributeError "'BinOp' object has no attribute 'value'".
