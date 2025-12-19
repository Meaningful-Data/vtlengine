import pandas as pd

from vtlengine import run, semantic_analysis


def main():
    script = """
        DS_r := DS_1[filter isnull(Me_dummy)];
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

    semantic_analysis(script=script, data_structures=data_structures)


if __name__ == "__main__":
    main()
