import pandas as pd

from vtlengine import run


def main():
    script = """
        DS_r:=DS_1;
    """

    data_structures = {
        'datasets': [
            {'name': 'DS_1',
             'DataStructure': [
                 {'name': 'Id_1',
                  'type': 'Time_Period',
                  'role': 'Identfier',
                  'nullable': False},
                 {'name': 'Me_1',
                  'type': 'String',
                  'role': 'Measure',
                  'nullable': True},
             ]
             }
        ]
    }

    data_df = pd.DataFrame(
        {"Id_1": [1, 2],
         "Me_1": ["ab123", "ab4b6"]})

    datapoints = {"DS_1": data_df}

    run_result = run(script=script, data_structures=data_structures,
                     datapoints=datapoints)

    result_data = run_result['DS_r'].data
    result_data.to_csv("development/data/random/dataPoints/output/DS_r.csv", index=False)

    print(run_result)


if __name__ == '__main__':
    main()
