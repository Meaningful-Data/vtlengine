###########
SDMX Inputs
###########

If you work mostly with SDMX, the engine has you covered. It reads
SDMX-ML, SDMX-JSON and SDMX-CSV files directly, understands the
``pysdmx`` structure objects (``Schema``, ``DataStructureDefinition``,
``Dataflow``) and ``PandasDataset`` bundles, and can produce SDMX
``TransformationScheme`` objects on the way out.

This page collects the SDMX-specific patterns you're likely to run
into beyond the basics: loading SDMX files through :meth:`vtlengine.run`,
binding one ``Dataflow`` to two VTL datasets, feeding the engine a
registered ``TransformationScheme``, and aliasing SDMX dataflows to the
names your script uses.

For the basic case — a script plus a list of ``PandasDataset`` objects
from ``pysdmx.io.get_datasets`` — start with :ref:`run-sdmx-section` in
the :doc:`walkthrough` instead. This page picks up where that leaves off.

*******************
Run with SDMX files
*******************

If you have SDMX files on disk and you'd rather not convert them
yourself, hand them straight to :meth:`vtlengine.run` — there's no need
to go through :meth:`vtlengine.run_sdmx` for that. The engine picks the
right loader from each file's extension and translates the contents to
its internal VTL representation behind the scenes.

Accepted SDMX formats for **data_structures**:

- SDMX-ML structure files (``.xml``)
- SDMX-JSON structure files (``.json``)
- pysdmx objects (``Schema``, ``DataStructureDefinition``, ``Dataflow``)

Accepted SDMX formats for **datapoints**:

- SDMX-ML data files (``.xml``)
- SDMX-JSON data files (``.json``)
- SDMX-CSV data files (``.csv``) — auto-detected

The engine routes each file based on its extension. For CSV files in
particular, it first tries to parse them as SDMX-CSV and falls back to
plain CSV if that doesn't work — so you can mix the two without
thinking about it.

One detail that catches users out: the dataset name in the structure
file (the DSD ID) often differs from the name in the data file (the
Dataflow reference). When that happens, use the ``sdmx_mappings``
argument to alias the data file's URN to whatever name your script
uses:

.. code-block:: python

    from pathlib import Path

    from vtlengine import run

    # Using SDMX structure and data files directly
    structure_file = Path("path/to/structure.xml")  # SDMX-ML structure
    data_file = Path("path/to/data.xml")            # SDMX-ML data

    # Map the data file's Dataflow URN to the structure's DSD name
    mapping = {"Dataflow=AGENCY:DATAFLOW_ID(1.0)": "DSD_NAME"}

    script = "DS_r <- DSD_NAME [calc Me_2 := OBS_VALUE * 2];"

    result = run(
        script=script,
        data_structures=structure_file,
        datapoints=data_file,
        sdmx_mappings=mapping
    )

You can also use ``sdmx_mappings`` to give datasets custom names in your
VTL script:

.. code-block:: python

    from pathlib import Path

    from vtlengine import run

    structure_file = Path("path/to/structure.xml")
    data_file = Path("path/to/data.xml")

    script = "DS_r <- MY_DATASET [calc Me_2 := OBS_VALUE * 2];"

    # Map SDMX URN to VTL dataset name
    mapping = {"Dataflow=MD:TEST_DF(1.0)": "MY_DATASET"}

    result = run(
        script=script,
        data_structures=structure_file,
        datapoints=data_file,
        sdmx_mappings=mapping
    )

You can also mix VTL JSON structures with SDMX structures and plain CSV
datapoints with SDMX data files:

.. code-block:: python

    from pathlib import Path

    from vtlengine import run

    # Mix of VTL JSON and SDMX structures
    vtl_structure = {"datasets": [{"name": "DS_1", "DataStructure": [...]}]}
    sdmx_structure = Path("path/to/sdmx_structure.xml")

    # Mix of plain CSV and SDMX data
    datapoints = {
        "DS_1": Path("path/to/plain_data.csv"),          # Plain CSV
        "DS_2": Path("path/to/sdmx_data.xml"),           # SDMX-ML
    }

    result = run(
        script=script,
        data_structures=[vtl_structure, sdmx_structure],
        datapoints=datapoints
    )


.. _sharing_one_dataflow_between_two_datasets:

*****************************************
Sharing one Dataflow between two datasets
*****************************************

A common SDMX situation: you have two datasets that share a single
``Dataflow`` (and therefore one ``DataStructureDefinition``) but hold
different data. Maybe two reporting periods, maybe a
previous-vs-current snapshot. You'd like both to appear in your VTL
script under separate names — without cloning the structure.

The trick is to call ``to_vtl_json`` on the same ``Dataflow`` twice,
each time with a different ``dataset_name``. That gives you two VTL
data structures pointing at the same SDMX backbone.

.. code-block:: python

    import pandas as pd
    from pysdmx.model.concept import Concept, DataType
    from pysdmx.model.dataflow import (
        Component, Components, Dataflow, DataStructureDefinition, Role,
    )

    from vtlengine import run
    from vtlengine.files.sdmx_handler import to_vtl_json


    def build_components() -> Components:
        return Components([
            Component(id="Id_1", required=True, role=Role.DIMENSION,
                      concept=Concept(id="Id_1", dtype=DataType.INTEGER)),
            Component(id="Me_1", required=False, role=Role.MEASURE,
                      concept=Concept(id="Me_1", dtype=DataType.FLOAT)),
        ])


    dataflow = Dataflow(
        id="DF_1", agency="ME", version="1.0",
        structure=DataStructureDefinition(
            id="DSD_1", agency="ME", version="1.0",
            components=build_components(),
        ),
    )

    data_structures = [
        to_vtl_json(dataflow, dataset_name="DS_1"),
        to_vtl_json(dataflow, dataset_name="DS_2"),
    ]

    datapoints = {
        "DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0]}),
        "DS_2": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.0, 25.0, 30.0]}),
    }

    script = """
        DS_diff  <- DS_2 - DS_1;
        DS_equal <- DS_1 = DS_2;
    """

    result = run(script=script, data_structures=data_structures,
                 datapoints=datapoints, return_only_persistent=False)

Expected output for ``DS_diff``::

     Id_1  Me_1
        1   0.0
        2   5.0
        3   0.0

Expected output for ``DS_equal``::

     Id_1  bool_var
        1      True
        2     False
        3      True


**********************************
TransformationScheme as the script
**********************************

If your VTL script already lives in an SDMX repository as a
``TransformationScheme``, you don't have to extract the text and pass
it as a string — ``run_sdmx`` accepts the object directly. Each
``Transformation`` inside the scheme contributes one statement to the
script the engine executes.

When you don't pass a ``mappings`` argument, the script must reference
a single input dataset, and the data file you load must contain just
one dataset too — the engine has to figure out the pairing
unambiguously.

.. code-block:: python

    from pathlib import Path

    from pysdmx.io import get_datasets
    from pysdmx.model.vtl import TransformationScheme, Transformation

    from vtlengine import run_sdmx

    data = Path("docs/_static/data.xml")
    structure = Path("docs/_static/metadata.xml")
    datasets = get_datasets(data, structure)
    script = TransformationScheme(
        id="TS1",
        version="1.0",
        agency="MD",
        vtl_version="2.1",
        items=[
            Transformation(
                id="T1",
                uri=None,
                urn=None,
                name=None,
                description=None,
                expression="DS_1 [calc Me_4 := OBS_VALUE];",
                is_persistent=True,
                result="DS_r1",
                annotations=(),
            ),
            Transformation(
                id="T2",
                uri=None,
                urn=None,
                name=None,
                description=None,
                expression="DS_1 [rename OBS_VALUE to Me_5];",
                is_persistent=True,
                result="DS_r2",
                annotations=(),
            )
        ],
    )
    run_sdmx(script, datasets=datasets)


*************************************
Mapping SDMX dataflows to VTL aliases
*************************************

Sometimes the name your script uses for a dataset doesn't match the
SDMX dataflow's short-URN — maybe the script was written first, maybe
the SDMX names are too unwieldy to drop into VTL expressions, or maybe
you just want a friendlier handle. Pass a ``mappings`` argument to
bridge the two.

You can express the mapping as a plain ``dict`` or as a ``pysdmx``
``VtlDataflowMapping`` object — pick whichever fits your code:

.. code-block:: python

    from pathlib import Path

    from pysdmx.io import get_datasets
    from pysdmx.model.vtl import TransformationScheme, Transformation
    from pysdmx.model.vtl import VtlDataflowMapping

    from vtlengine import run_sdmx

    data = Path("docs/_static/data.xml")
    structure = Path("docs/_static/metadata.xml")
    datasets = get_datasets(data, structure)
    script = TransformationScheme(
        id="TS1",
        version="1.0",
        agency="MD",
        vtl_version="2.1",
        items=[
            Transformation(
                id="T1",
                uri=None,
                urn=None,
                name=None,
                description=None,
                expression="DS_1 [calc Me_4 := OBS_VALUE]",
                is_persistent=True,
                result="DS_r",
                annotations=(),
            ),
        ],
    )
    # Mapping using VtlDataflowMapping object:
    mapping = VtlDataflowMapping(
            dataflow="urn:sdmx:org.sdmx.infomodel.datastructure.Dataflow=MD:TEST_DF(1.0)",
            dataflow_alias="DS_1",
            id="VTL_MAP_1",
        )

    # Mapping using dictionary:
    mapping = {
    "Dataflow=MD:TEST_DF(1.0)": "DS_1"
    }
    run_sdmx(script, datasets, mappings=mapping)


Files used in the examples can be found here:

- :download:`data.xml <_static/data.xml>`
- :download:`metadata.xml <_static/metadata.xml>`
