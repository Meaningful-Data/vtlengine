########################
10 minutes to VTL Engine
########################

Summarizes the main functions of the VTL Engine

*****************
Semantic Analysis
*****************
To perform the validation of a VTL script, please use the semantic_analysis function.
Here is an example:

.. code-block:: python

    from vtlengine import semantic_analysis
    from pathlib import Path

    base_path = Path(__file__).parent / "tests/API/data/"
    script = base_path / Path("vtl/1.vtl")
    datastructures = base_path / Path("DataStructure/input")
    value_domains = base_path / Path("ValueDomain/VD_1.json")
    external_routines = base_path / Path("sql/1.sql")

    semantic_analysis(script=script, data_structures=datastructures,
                      value_domains=value_domains, external_routines=external_routines)

.. code-block:: python

The semantic analysis function will return a dictionary of the computed datasets and their structure.

*****************
Run VTL Scripts
*****************

To execute a VTL script, please use the run function. Here is an example:

.. code-block:: python

    from vtlengine import run
    from pathlib import Path

    base_path = Path(__file__).parent / "tests/API/data/"
    script = base_path / Path("vtl/1.vtl")
    datastructures = base_path / Path("DataStructure/input")
    datapoints = base_path / Path("DataSet/input")
    output_folder = base_path / Path("DataSet/output")

    value_domains = None
    external_routines = None

    run(script=script, data_structures=datastructures, datapoints=datapoints,
        value_domains=value_domains, external_routines=external_routines,
        output_folder=output_folder, return_only_persistent=True
        )

The VTL engine will load each datapoints file as being needed, reducing the memory footprint.
When the output parameter is set, the engine will write the result of the computation
to the output folder, else it will include the data in the dictionary of the computed datasets.

For more information on usage, please refer to the `API documentation <https://docs.vtlengine.meaningfuldata.eu/api.html>`_
