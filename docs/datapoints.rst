###########
Datapoints
###########

Datapoints are the actual rows of your data — the bit you'd see in a
spreadsheet, a CSV file, or an SDMX data message. While
:doc:`data_structures` tell the engine *what your data looks like* (the
columns and their types), datapoints provide the values that actually
flow through your transformations.

How you hand them to the engine depends on which method you're using.


********************************
For :meth:`vtlengine.run`
********************************

The ``datapoints`` argument is most often a Python ``dict`` where each
key is the dataset name and each value is the data for that dataset:

.. code-block:: python

    import pandas as pd
    from pathlib import Path
    from vtlengine import run

    datapoints = {
        "DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10, 20, 30]}),
        "DS_2": Path("data/DS_2.csv"),
        "DS_3": Path("data/DS_3.xml"),  # SDMX-ML
    }

The key must match the ``name`` of the corresponding dataset in your
:doc:`data_structures`. Inside the dict, each value can take several
shapes:

* a **pandas DataFrame**, when your data is already in memory
* a **Path to a CSV file** (plain or SDMX-CSV — extension and content
  decide which loader to use)
* a **Path to an SDMX-ML (``.xml``) data file**

.. note::
    SDMX-JSON is supported for **structures**, not for data. For data,
    use SDMX-ML or SDMX-CSV.

For the exact SDMX versions supported on the data side (SDMX-CSV 1.0 /
2.0 / 2.1, SDMX-ML 2.1 / 3.0 / 3.1), see pysdmx's
`Formats and versions supported
<https://py.sdmx.io/api/io/general_reader.html#formats-and-versions-supported>`_.

You can mix them freely — one dataset can come from a DataFrame, another
from a plain CSV, a third from an SDMX-ML file. The engine routes each
value to the right loader.

.. code-block:: python

    datapoints = {
        "DS_1": Path("plain.csv"),       # plain CSV
        "DS_2": Path("sdmx_data.xml"),   # SDMX-ML
        "DS_3": Path("sdmx_data.csv"),   # SDMX-CSV (with embedded structure)
    }

Whichever shape you pick, the column names in the data must match the
component names declared in the data structure. Type conversion is
handled by the engine according to the declared component types.


Shortcut: a single Path
=======================

If you don't want to build a dict, you can pass a single ``Path`` —
either to a file or to a directory of files. In that case, the filename
(without extension) becomes the dataset name. ``Path("data/DS_1.csv")``
is loaded as ``"DS_1"``.


Working with S3 storage
=======================

S3 URIs (``s3://bucket/path/to/data.csv``) are accepted as ``datapoints``
values, but only when the DuckDB backend is enabled
(``use_duckdb=True``). See :doc:`duckdb_engine` for the AWS environment
variables involved.


********************************
For :meth:`vtlengine.run_sdmx`
********************************

For ``run_sdmx`` there's no separate ``datapoints`` argument. Each
``pysdmx`` ``PandasDataset`` you pass already carries its rows in
``.data`` (a DataFrame) alongside its structure in ``.structure`` (a
``Schema``). That pairing is the whole point of using ``run_sdmx``:

.. code-block:: python

    from pathlib import Path
    from pysdmx.io import get_datasets
    from vtlengine import run_sdmx

    datasets = get_datasets(Path("data.xml"), Path("structure.xml"))
    run_sdmx(script, datasets)

If your SDMX source returns more datasets than the script needs, filter
the list before passing it — the engine errors out on extras. See the
:ref:`Run SDMX <run-sdmx-section>` section of the :doc:`walkthrough` for
the rule and how mappings interact with it.


.. seealso::

    * :doc:`data_structures` — what describes the shape of each dataset.
    * :doc:`vtl_scripts` — how the script references these datasets by name.
    * :doc:`sdmx_inputs` — SDMX-specific patterns including ``sdmx_mappings``.
    * :doc:`duckdb_engine` — S3 URI support.
