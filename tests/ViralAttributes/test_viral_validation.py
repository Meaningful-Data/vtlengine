import pandas as pd
import pytest

from vtlengine import run
from vtlengine.Exceptions import SemanticError

DATA_STRUCTURES = {
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
DATAPOINTS = {"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0]})}


class TestViralPropagationSemanticErrors:
    def test_duplicate_variable_rule_raises_error(self):
        script = """
            define viral propagation rule1 (variable At_1) is
                when "C" then "C"
            end viral propagation;
            define viral propagation rule2 (variable At_1) is
                when "N" then "N"
            end viral propagation;
            DS_r <- DS_1;
        """
        with pytest.raises(SemanticError, match="1-3-3-1"):
            run(script=script, data_structures=DATA_STRUCTURES, datapoints=DATAPOINTS)

    def test_duplicate_enumeration_raises_error(self):
        script = """
            define viral propagation dup (variable At_1) is
                when "C" then "C";
                when "C" then "N"
            end viral propagation;
            DS_r <- DS_1;
        """
        with pytest.raises(SemanticError, match="1-3-3-4"):
            run(script=script, data_structures=DATA_STRUCTURES, datapoints=DATAPOINTS)
