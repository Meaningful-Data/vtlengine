import pandas as pd

from vtlengine import run


def main():
    #127
    #script = """
    #    DS_A := datediff("2020-12-14","2021-04-20");
    #"""

    #456
    #script = """
    #    DS_A := datediff("2022Q1", "2023Q2");
    #"""

    #127
    script = """
        DS_A := datediff("2021Q2","2021-11-04");
    """

    # 4,5,4
    #script = """
    #     DS_A := DS_1[calc Me_3 := datediff(Me_1, Me_2)];
    #"""

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
                  'type': 'Date',
                  'role': 'Measure',
                  'nullable': True},
                 {'name': 'Me_2',
                  'type': 'Date',
                  'role': 'Measure',
                  'nullable': True}
             ]
             }
        ]
    }

    data_df = pd.DataFrame(
        {"Id_1": [1, 2, 3],
         "Me_1": ["2020-01-05", "2020-02-06", "2020-03-05"],
         "Me_2":  ["2020-01-01", "2020-02-01", "2020-03-01"]})

    datapoints = {"DS_1": data_df}

    run_result = run(script=script, data_structures=data_structures,
                     datapoints=datapoints)

    print(run_result['DS_A'])


if __name__ == '__main__':
    main()