"""Viral propagation example: a `sum` aggregation carrying viral attributes.

Viral attributes are attributes flagged to survive ("go viral" through) operations
that would normally drop attributes. A `define viral propagation` rule says *how* the
attribute values are combined when rows collapse -- here, during a `sum(... group by ...)`
aggregation.

Two rules are shown:

* `SUM_RULE`  -> `aggregate sum` on the numeric viral attribute `At_num`. The attribute
  values of each group are summed, exactly like the measure.
* `STR_RULE`  -> an enumerated (when/then/else) rule on the string viral attribute
  `At_str`. `aggregate sum` is a *numeric* combiner (it would raise a TypeError on
  strings), so a string attribute is propagated with an enumerated rule instead.
"""

import pandas as pd

from vtlengine import run


def main() -> None:
    script = """
        define viral propagation SUM_RULE (variable At_num) is
            aggregate sum
        end viral propagation;

        define viral propagation STR_RULE (variable At_str) is
            when "C" then "C";
            when "N" then "N";
            else "F"
        end viral propagation;

        DS_r <- sum(DS_1 group by Id_1);
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    # Numeric viral attribute -> combined with `aggregate sum`.
                    {"name": "At_num", "type": "Integer", "role": "Viral Attribute", "nullable": True},
                    # String viral attribute -> combined with the enumerated STR_RULE.
                    {"name": "At_str", "type": "String", "role": "Viral Attribute", "nullable": True},
                ],
            }
        ]
    }

    data_df = pd.DataFrame(
        {
            "Id_1": [1, 1, 2],
            "Id_2": [1, 2, 1],
            "Me_1": [10.0, 20.0, 30.0],
            "At_num": [5, 7, 3],
            "At_str": ["C", "N", "C"],
        }
    )

    datapoints = {"DS_1": data_df}

    run_result = run(script=script, data_structures=data_structures, datapoints=datapoints)

    # Expected DS_r (grouped by Id_1):
    #   Id_1  Me_1  At_num  At_str
    #      1  30.0      12       C   # Me_1: 10+20, At_num: 5+7 (sum), At_str: {C,N} -> "C"
    #      2  30.0       3       C   # single row: values pass through unchanged
    print(run_result["DS_r"].data.to_string(index=False))


if __name__ == "__main__":
    main()
