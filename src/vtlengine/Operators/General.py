import re
from typing import Any, Dict, List, Union

import duckdb

from vtlengine.DataTypes import COMP_NAME_MAPPING
from vtlengine.Exceptions import RunTimeError, SemanticError
from vtlengine.Model import Component, Dataset, ExternalRoutine, Role, Scalar, names_equal
from vtlengine.Operators import Binary, Unary
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


class Membership(Binary):
    """Membership operator class.

    It inherits from Binary class and has the following class methods:

    Class methods:
        Validate: Checks if str right operand is actually within the Dataset.
        Evaluate: Checks validate operation and return the dataset to perform it.
    """

    @classmethod
    def validate(cls, left_operand: Any, right_operand: Any) -> Union[Dataset, Scalar]:
        dataset_name = VirtualCounter._new_ds_name()
        if right_operand not in left_operand.components:
            raise SemanticError(
                "1-1-1-10",
                op=cls.op,
                comp_name=right_operand,
                dataset_name=left_operand.name,
            )

        component = left_operand.components[right_operand]
        if len(left_operand.get_identifiers()) == 0:
            return Scalar(name=dataset_name, data_type=component.data_type, value=None)

        promote_to_measure = component.role in (Role.IDENTIFIER, Role.ATTRIBUTE)
        result_components = {
            name: comp
            for name, comp in left_operand.components.items()
            if comp.role == Role.IDENTIFIER
            or comp.role == Role.VIRAL_ATTRIBUTE
            or (not promote_to_measure and names_equal(comp.name, right_operand))
        }
        if promote_to_measure:
            measure_name = COMP_NAME_MAPPING[component.data_type]
            result_components[measure_name] = Component(
                name=measure_name,
                data_type=component.data_type,
                role=Role.MEASURE,
                nullable=component.nullable,
            )
        return Dataset(name=dataset_name, components=result_components, data=None)


class Alias(Binary):
    """Alias operator class
    It inherits from Binary class, and has the following class methods:

    Class methods:
        Validate: Ensures the name given in the right operand is different from the
        name of the Dataset. Evaluate: Checks if the data between both operators are the same.
    """

    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: Union[str, Dataset]) -> Dataset:
        new_name = right_operand if isinstance(right_operand, str) else right_operand.name
        if new_name != left_operand.name and new_name in left_operand.get_components_names():
            raise SemanticError("1-3-1", alias=new_name)
        return Dataset(name=new_name, components=left_operand.components, data=None)


class Eval(Unary):
    """Eval operator class
    It inherits from Unary class and has the following class methods

    Class methods:
        Validate: checks if the external routine name is the same as the operand name,
        which must be a Dataset.
        Evaluate: Checks if the operand and the output is actually a Dataset.

    """

    @staticmethod
    def _execute_query(
        query: str,
        dataset_names: List[str],
        schemas: Dict[str, Dict[str, Component]],
    ) -> List[str]:
        """Validate the external SQL against the operand schemas and return the result columns.

        Creates empty typed tables for each operand in an in-memory DuckDB connection,
        runs the query, and returns the column names DuckDB produces. No data flows
        through; this is a schema-validation pass.
        """
        query = re.sub(r'"([^"]*)"', r"'\1'", query)
        for forbidden in ["INSTALL", "LOAD"]:
            if re.search(rf"\b{forbidden}\b", query, re.IGNORECASE):
                raise SemanticError("1-1-1-21", command=forbidden)
        if re.search(r"FROM\s+'https?://", query, re.IGNORECASE):
            raise SemanticError("1-1-1-22")
        try:
            conn = duckdb.connect(database=":memory:", read_only=False)
            conn.execute("SET enable_external_access = false")
            conn.execute("SET allow_unsigned_extensions = false")
            conn.execute("SET allow_community_extensions = false")
            conn.execute("SET autoinstall_known_extensions = false")
            conn.execute("SET autoload_known_extensions = false")
            conn.execute("SET lock_configuration = true")

            # Lazy import to avoid a circular dependency between Operators and the
            # duckdb_transpiler.io package (which transitively imports files.sdmx_handler).
            from vtlengine.duckdb_transpiler.io._validation import build_create_table_sql

            try:
                for ds_name in dataset_names:
                    conn.execute(build_create_table_sql(ds_name, schemas[ds_name]))
                result = conn.execute(query)
                column_names = [col[0] for col in result.description or []]
                conn.close()
            except Exception as e:
                conn.close()
                raise RunTimeError("2-1-1-1", op="eval", error=e)
        except RunTimeError:
            raise
        except Exception as e:
            raise RunTimeError("2-1-1-1", op="eval", error=e)
        return column_names

    @classmethod
    def validate(  # type: ignore[override]
        cls,
        operands: Dict[str, Dataset],
        external_routine: ExternalRoutine,
        output: Dataset,
    ) -> Dataset:
        schemas: Dict[str, Dict[str, Component]] = {}
        for ds_name in external_routine.dataset_names:
            if ds_name not in operands:
                raise ValueError(
                    f"External Routine dataset {ds_name} is not present in Eval operands"
                )
            schemas[ds_name] = operands[ds_name].components

        component_names = cls._execute_query(
            external_routine.query, external_routine.dataset_names, schemas
        )
        for comp_name in component_names:
            if comp_name not in output.components:
                raise SemanticError(
                    "1-1-1-10", op=cls.op, comp_name=comp_name, dataset_name=output.name
                )

        for comp_name in output.components:
            if comp_name not in component_names:
                raise ValueError(f"Component {comp_name} not found in External Routine result")

        output.name = external_routine.name

        return output
