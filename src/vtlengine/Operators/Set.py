from typing import Any, Dict, List

from vtlengine.connection import con
from vtlengine.DataTypes import binary_implicit_promotion
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Dataset
from vtlengine.Operators import Operator
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


class Set(Operator):
    @classmethod
    def check_same_structure(cls, dataset_1: Dataset, dataset_2: Dataset) -> None:
        if len(dataset_1.components) != len(dataset_2.components):
            raise SemanticError(
                "1-1-17-1",
                op=cls.op,
                dataset_1=dataset_1.name,
                dataset_2=dataset_2.name,
            )

        for comp in dataset_1.components.values():
            if comp.name not in dataset_2.components:
                raise Exception(f"Component {comp.name} not found in dataset {dataset_2.name}")
            second_comp = dataset_2.components[comp.name]
            binary_implicit_promotion(
                comp.data_type,
                second_comp.data_type,
                cls.type_to_check,
                cls.return_type,
            )
            if comp.role != second_comp.role:
                raise Exception(
                    f"Component {comp.name} has different roles "
                    f"in datasets {dataset_1.name} and {dataset_2.name}"
                )

    @classmethod
    def validate(cls, operands: List[Dataset]) -> Dataset:
        base_operand = operands[0]
        for operand in operands[1:]:
            cls.check_same_structure(base_operand, operand)

        result_components: Dict[str, Any] = {}
        for operand in operands:
            if len(result_components) == 0:
                result_components = operand.components
            else:
                for comp_name, comp in operand.components.items():
                    current_comp = result_components[comp_name]
                    result_components[comp_name].data_type = binary_implicit_promotion(
                        current_comp.data_type, comp.data_type
                    )
                    result_components[comp_name].nullable = current_comp.nullable or comp.nullable

        result = Dataset(name="result", components=result_components, data=None)
        return result


class Union(Set):
    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        queries = []
        id_names = operands[0].get_identifiers_names()
        ids_str = ", ".join(id_names)
        col_names = list(operands[0].data.columns)
        cols_str = ", ".join(col_names)

        for i in range(len(operands)):
            name = VirtualCounter._new_temp_view_name()
            con.register(name, operands[i].data)
            queries.append(f"SELECT {cols_str}, {i} AS priority FROM {name}")

        union_query = " UNION ALL ".join(queries)
        vds_union_int = VirtualCounter._new_temp_view_name()
        vds_union_final = VirtualCounter._new_temp_view_name()

        final_query = f"""
            WITH {vds_union_final} AS (
                {union_query}
            ),
            {vds_union_int} AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY {ids_str} ORDER BY priority ASC) AS rn
                FROM {vds_union_final}
            )
            SELECT {cols_str}
            FROM {vds_union_int}
            WHERE rn = 1
        """

        result.data = con.query(final_query)
        return result


class Intersection(Set):
    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)

        col_names = list(operands[0].data.columns)
        cols_str = ", ".join(col_names)

        list_names = []
        for ds in operands:
            temp_name = VirtualCounter._new_temp_view_name()
            con.register(temp_name, ds.data)
            list_names.append(temp_name)

        sub_query = [f"SELECT {cols_str} FROM {name}" for name in list_names]
        query = " INTERSECT ".join(sub_query)

        result.data = con.query(query)
        return result


class Symdiff(Set):
    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)

        id_names = operands[0].get_identifiers_names()
        ids_str = ", ".join(id_names)

        ds1, ds2 = operands

        name1 = VirtualCounter._new_temp_view_name()
        name2 = VirtualCounter._new_temp_view_name()

        con.register(name1, ds1.data)
        con.register(name2, ds2.data)

        col_names = list(operands[0].data.columns)
        cols_str = ", ".join(col_names)

        vds_symdiff_final = VirtualCounter._new_temp_view_name()
        vds_symdiff_int = VirtualCounter._new_temp_view_name()

        diff1 = f"""
                    SELECT {name1}.*, 1 AS priority
                    FROM {name1}
                    ANTI JOIN {name2} USING ({",".join([f"{col}" for col in id_names])})
                """

        diff2 = f"""
                    SELECT {name2}.*, 2 AS priority
                    FROM {name2}
                    ANTI JOIN {name1} USING ({",".join([f"{col}" for col in id_names])})
                """

        final_query = f"""
                    WITH {vds_symdiff_final} AS (
                        {diff1}
                        UNION ALL
                        {diff2}
                    ),
                    {vds_symdiff_int} AS (
                        SELECT *,
                            ROW_NUMBER() OVER (PARTITION BY {ids_str} ORDER BY priority ASC) AS rn
                        FROM {vds_symdiff_final}
                    )
                    SELECT {cols_str}
                    FROM {vds_symdiff_int}
                    WHERE rn = 1
                """

        result.data = con.query(final_query)
        return result


class Setdiff(Set):
    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)

        ds1, ds2 = operands
        name1 = VirtualCounter._new_temp_view_name()
        name2 = VirtualCounter._new_temp_view_name()

        con.register(name1, ds1.data)
        con.register(name2, ds2.data)

        query = f"""
            SELECT * FROM {name1}
            EXCEPT
            SELECT * FROM {name2}
        """
        result.data = con.query(query)
        return result
