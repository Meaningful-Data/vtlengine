import pandas as pd

from vtlengine import run


def main():
    script = """
        DS_A <- DS_1 * 10;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10, 20, 30]})

    datapoints = {"DS_1": data_df}

    # Run with pandas (default)
    run_result = run(script=script, data_structures=data_structures, datapoints=datapoints)
    print("Pandas result:", run_result)

    # Run with DuckDB
    run_result_duckdb = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        use_duckdb=True,
    )
    print("DuckDB result:", run_result_duckdb)


if __name__ == "__main__":
    main()
