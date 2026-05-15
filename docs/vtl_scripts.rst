############
VTL Scripts
############

A VTL script is just text — usually a handful of lines telling the
engine what transformations to apply to your data. You write it once,
hand it to :meth:`vtlengine.run` or :meth:`vtlengine.run_sdmx`, and the
engine parses it, type-checks it against your data structures, and
executes it.

This page is about the *engine side* of scripts: which forms the engine
accepts, how persistent and non-persistent results work, and the related
options. For the VTL language itself — what operators exist, how the
syntax is spelled — see the
`VTL 2.1 Reference Manual
<https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf>`_.


******************************
Three ways to provide a script
******************************

You can hand the engine your script in three different shapes, depending
on where it came from.

**1. As a Python string.** The most direct option — paste your VTL right
into your Python code:

.. code-block:: python

    script = "DS_r <- DS_1 [calc Me_2 := Me_1 * 10];"

Handy for quick experiments and tests.

**2. As a file path.** Once your scripts grow beyond a couple of lines,
you'll want them in their own ``.vtl`` files. Pass a ``pathlib.Path``:

.. code-block:: python

    from pathlib import Path
    script = Path("my_transformations.vtl")

**3. As a pysdmx TransformationScheme.** If your script is already
registered in an SDMX repository, you can hand the engine the
``TransformationScheme`` directly. Each ``Transformation`` inside the
scheme contributes one statement to the executed script.

.. code-block:: python

    from pysdmx.model.vtl import TransformationScheme, Transformation
    script = TransformationScheme(id="TS1", version="1.0", agency="MD",
                                  vtl_version="2.1", items=[...])

See :doc:`sdmx_inputs` for a full example.

All three forms are accepted by :meth:`vtlengine.run`,
:meth:`vtlengine.run_sdmx`, :meth:`vtlengine.semantic_analysis`,
:meth:`vtlengine.prettify`, and :meth:`vtlengine.generate_sdmx`.


****************************************
Persistent vs non-persistent results
****************************************

VTL has two assignment operators, and the distinction matters when you
collect the results:

* ``<-`` — **persistent assignment**. The dataset on the left is a final
  result of the script, something you want back.
* ``:=`` — **non-persistent assignment**. The dataset on the left is an
  intermediate stepping stone; you don't necessarily care about its
  value, only that it helps compute something else.

By default, :meth:`vtlengine.run` and :meth:`vtlengine.run_sdmx` only
hand back the persistent results — usually that's what you want. If
you'd rather see every dataset the script produced (e.g. while
debugging), pass ``return_only_persistent=False``:

.. code-block:: python

    script = """
        DS_tmp := DS_1 * 10;          /* non-persistent */
        DS_r   <- DS_tmp + DS_2;      /* persistent */
    """

    # Default: returns only DS_r
    result = run(script=script, data_structures=..., datapoints=...)

    # Returns both DS_tmp and DS_r
    result = run(
        script=script, data_structures=..., datapoints=...,
        return_only_persistent=False,
    )


***************************
Cleaning up the formatting
***************************

If your script came from a manual, a copy-paste, or a string you patched
together programmatically, it might not be the easiest thing to read.
:meth:`vtlengine.prettify` reformats it:

.. code-block:: python

    from vtlengine import prettify
    print(prettify("DS_r<-DS_1[calc Me_2:=Me_1*10];"))

See the :ref:`Prettify <prettify>` section of the :doc:`walkthrough` for
a complete example.


*******************************
Going the other way: VTL → SDMX
*******************************

If you've written a VTL script and want to register it with an SDMX
repository, :meth:`vtlengine.generate_sdmx` does the inverse of
accepting a ``TransformationScheme`` as input: it parses the script and
produces a pysdmx ``TransformationScheme`` object you can serialise to
SDMX-ML.

.. seealso::

    * :doc:`sdmx_inputs` — passing a ``TransformationScheme`` to
      :meth:`vtlengine.run_sdmx`.
    * :doc:`api` — full reference for :meth:`vtlengine.semantic_analysis`,
      :meth:`vtlengine.prettify`, and :meth:`vtlengine.generate_sdmx`.
