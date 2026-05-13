###########
SDMX Inputs
###########

The VTL Engine integrates natively with SDMX. :meth:`vtlengine.run` can
load SDMX structure files and ``pysdmx`` structure objects directly — you
do not need :meth:`vtlengine.run_sdmx` for that. Use ``run_sdmx`` only
when structure and data already travel together inside ``pysdmx``
``PandasDataset`` objects.

For the basic ``run_sdmx`` call (a script + a list of ``PandasDataset``
objects from ``pysdmx.io.get_datasets``), see :ref:`run-sdmx-section` in
the :doc:`walkthrough`. This page collects the more advanced SDMX-flavored
patterns.

***********************
Run with SDMX files
***********************

:meth:`vtlengine.run` can load SDMX files directly, without using
:meth:`vtlengine.run_sdmx`. This provides a seamless workflow for SDMX
data without requiring manual conversion to VTL JSON format.

Supported SDMX formats for **data_structures**:

- SDMX-ML structure files (``.xml``)
- SDMX-JSON structure files (``.json``)
- pysdmx objects (``Schema``, ``DataStructureDefinition``, ``Dataflow``)

Supported SDMX formats for **datapoints**:

- SDMX-ML data files (``.xml``)
- SDMX-JSON data files (``.json``)
- SDMX-CSV data files (``.csv``) — with automatic detection

SDMX files are automatically detected by their extension. For CSV files,
the engine first attempts to parse as SDMX-CSV, then falls back to plain
CSV if SDMX parsing fails.

When using SDMX files, the dataset name in the structure file (from the
DataStructureDefinition ID) may differ from the name in the data file
(from the Dataflow reference). Use the ``sdmx_mappings`` parameter to map
the data file's URN to the VTL dataset name used in your script:

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

******************************************
Sharing one Dataflow between two datasets
******************************************

A common SDMX pattern is having two datasets that share a single
``Dataflow`` (and therefore one ``DataStructureDefinition``) but contain
different data — for example, two reporting periods or a previous-vs-current
snapshot. The same ``Dataflow`` object can be passed to ``to_vtl_json``
twice with different ``dataset_name`` arguments to bind it to two VTL
aliases without cloning the structure.

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


***********************************
TransformationScheme as the script
***********************************

As part of its compatibility with ``pysdmx``, ``run_sdmx`` can take a
``TransformationScheme`` object as input. If no mapping is provided, the
VTL script must have a single input, and the data file must contain only
one dataset.

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


**************************************
Mapping SDMX dataflows to VTL aliases
**************************************

Mapping information can be used to link an SDMX input dataset to a VTL
input dataset via the ``VtlDataflowMapping`` object from ``pysdmx`` or a
dictionary.

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
