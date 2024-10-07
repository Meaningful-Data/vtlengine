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

    script = """
        DS_A := DS_1 * 10;
    """

    data_structures = {
        'datasets': [
            {'name': 'DS_1',
             'DataStructure': [
                 {'name': 'Id_1',
                  'type':
                  'Integer',
                  'role': 'Identifier',
                  'nullable': False},
                  {'name': 'Me_1',
                   'type': 'Number',
                   'role': 'Measure',
                   'nullable': True}
                   ]
            }
        ]
    }

    sa_result = semantic_analysis(script=script, data_structures=data_structures)

    print(sa_result)


The semantic analysis function will return a dictionary of the computed datasets and their structure.

*****************
Run VTL Scripts
*****************

To execute a VTL script, please use the run function. Here is an example:

.. code-block:: python

    from vtlengine import run
    import pandas as pd

    script = """
        DS_A := DS_1 * 10;
    """

    data_structures = {
        'datasets': [
            {'name': 'DS_1',
             'DataStructure': [
                 {'name': 'Id_1',
                  'type':
                  'Integer',
                  'role': 'Identifier',
                  'nullable': False},
                  {'name': 'Me_1',
                   'type': 'Number',
                   'role': 'Measure',
                   'nullable': True}
                   ]
            }
        ]
    }

    data_df = pd.DataFrame(
        {"Id_1": [1,2,3],
         "Me_1": [10, 20, 30]})

    datapoints = {"DS_1": data_df}

    run_result = run(script=script, data_structures=data_structures,
                    datapoints=datapoints)

    print(run_result)

The VTL engine will load each datapoints file as being needed, reducing the memory footprint.
When the output parameter is set, the engine will write the result of the computation
to the output folder, else it will include the data in the dictionary of the computed datasets.

For more information on usage, please refer to the `API documentation <https://docs.vtlengine.meaningfuldata.eu/api.html>`_
