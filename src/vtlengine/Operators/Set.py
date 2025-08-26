import copy
from typing import Any, Dict, List

from vtlengine.connection import con
from vtlengine.DataTypes import binary_implicit_promotion
from vtlengine.duckdb.duckdb_utils import empty_relation
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
        for operand in operands:
            operand.data = (
                operand.data
                if operand.data is not None
                else empty_relation(cols=list(operand.components.keys()))
            )
        queries = []
        id_names = operands[0].get_identifiers_names()
        ids_str = ", ".join(id_names)
        col_names = list(operands[0].components.keys())
        cols_str = ", ".join(col_names)

        for i, operand in enumerate(operands):
            name = VirtualCounter._new_temp_view_name()
            con.register(name, operand.data)
            queries.append(f"SELECT {cols_str}, {i} AS priority FROM {name}")

        union_query = " UNION ALL ".join(queries)
        vds_union_int = VirtualCounter._new_temp_view_name()
        vds_union_final = VirtualCounter._new_temp_view_name()

        # Here we create a final query that will select the first row for each identifier
        # based on the priority assigned in the previous step.
        # This ensures that we get a single row for each identifier in the final result.
        # The priority is used to determine which row to keep in case of duplicates.
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

        id_names = operands[0].get_identifiers_names()
        ids_str = ", ".join(id_names)
        col_names = list(operands[0].components.keys())

        cols_str = ", ".join(col_names)

        queries = []

        ref_name = None

        for i, operand in enumerate(operands):
            name = VirtualCounter._new_temp_view_name()
            con.register(name, operand.data)
            if ref_name is None:
                ref_name = copy.copy(name)
                continue

            query = f"""
                SELECT {ref_name}.*, {i} AS priority
                FROM {ref_name}
                INNER JOIN {name} USING ({",".join([f"{col}" for col in id_names])})
            """
            ref_name = copy.copy(name)
            queries.append(query)

        # We use here a UNION ALL to ensure that we keep all rows from the intersection
        # even if they have the same identifiers but different values in other columns.
        intersect_queries = " UNION ALL ".join(queries)

        vds_intersection_final = VirtualCounter._new_temp_view_name()
        vds_intersection_int = VirtualCounter._new_temp_view_name()

        # Here we create a final query that will select the first row for each identifier
        # based on the priority assigned in the previous step.
        # This ensures that we get a single row for each identifier in the final result.
        # The priority is used to determine which row to keep in case of duplicates.
        final_query = f"""
            WITH {vds_intersection_final} AS (
                {intersect_queries}
            ),
            {vds_intersection_int} AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY {ids_str} ORDER BY priority ASC) AS rn
                FROM {vds_intersection_final}
            )
            SELECT {cols_str}
            FROM {vds_intersection_int}
            WHERE rn = 1
        """

        result.data = con.query(final_query)
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

        final_query = f"""
            (
                SELECT {name1}.* FROM {name1}
                ANTI JOIN {name2}
                USING ({ids_str})
            )
            UNION ALL
            (
                SELECT {name2}.* FROM {name2}
                ANTI JOIN {name1}
                USING ({ids_str})
            )
        """

        result.data = con.query(final_query)
        return result


class Setdiff(Set):
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

        diff = f"""
                    SELECT {name1}.*
                    FROM {name1}
                    ANTI JOIN {name2} USING ({ids_str})
                """

        result.data = con.query(diff)
        return result
