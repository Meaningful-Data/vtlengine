from copy import copy
from typing import Any, Optional, Union

import duckdb
import pandas as pd
from duckdb.duckdb import DuckDBPyRelation

from vtlengine.AST.Grammar.tokens import (
    AND,
    CEIL,
    EQ,
    FLOOR,
    GT,
    GTE,
    LT,
    LTE,
    NEQ,
    OR,
    ROUND,
    XOR,
)
from vtlengine.connection import con
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    SCALAR_TYPES_CLASS_REVERSE,
    ScalarType,
    binary_implicit_promotion,
    check_binary_implicit_promotion,
    check_unary_implicit_promotion,
    unary_implicit_promotion,
)
from vtlengine.DataTypes.TimeHandling import (
    PERIOD_IND_MAPPING,
    TimeIntervalHandler,
    TimePeriodHandler,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar, ScalarSet
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.Utils._to_sql import LEFT, MIDDLE, TO_SQL_TOKEN
from vtlengine.Utils.duckdb_utils import duckdb_concat, duckdb_merge

ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]

# This allows changing the data type of the Measure in the result Data Set
# when the operator is applied to mono-measure Data Sets.
# TODO: Check if there are more operators that allow this
MONOMEASURE_CHANGED_ALLOWED = [CEIL, FLOOR, ROUND]
BINARY_COMPARISON_OPERATORS = [EQ, NEQ, GT, GTE, LT, LTE]
BINARY_BOOLEAN_OPERATORS = [AND, OR, XOR]

only_semantic = False


DUCKDB_RETURN_TYPES = Union[str, int, float, bool, None]
TIME_TYPES = ["TimeInterval", "TimePeriod", "Duration"]


def apply_unary_op(op: Any, me_name: str, value: Any) -> DUCKDB_RETURN_TYPES:
    op_token = TO_SQL_TOKEN.get(op, op)
    return f'{op_token}({me_name}) AS "{value}"'


def apply_bin_op(op: Any, me_name: str, left: Any, right: Any) -> DUCKDB_RETURN_TYPES:
    token_position = MIDDLE
    op_token = TO_SQL_TOKEN.get(op, op)
    if isinstance(op_token, tuple):
        op_token, token_position = op_token

    if token_position == LEFT:
        return f'{op_token}({left}, {right}) AS "{me_name}"'
    return f'({left} {op_token} {right}) AS "{me_name}"'


def _cast_time_types(data_type: Any, value: Any) -> str:
    if data_type.__name__ == "TimeInterval":
        return TimeIntervalHandler.from_iso_format(value)
    elif data_type.__name__ == "TimePeriod":
        return TimePeriodHandler(value)
    elif data_type.__name__ == "Duration":
        if value not in PERIOD_IND_MAPPING:
            raise Exception(f"Duration {value} is not valid")
        return PERIOD_IND_MAPPING[value]
    return str(value)


def cast_time_types_scalar(op: str, data_type: ScalarType, value: str) -> str:
    if op not in BINARY_COMPARISON_OPERATORS:
        return value
    if data_type.__name__ == "TimeInterval":
        return TimeIntervalHandler.from_iso_format(value)
    elif data_type.__name__ == "TimePeriod":
        return TimePeriodHandler(value)
    elif data_type.__name__ == "Duration":
        if value not in PERIOD_IND_MAPPING:
            raise Exception(f"Duration {value} is not valid")
        return PERIOD_IND_MAPPING[value]
    return value


def _handle_str_number(x: Union[str, int, float]) -> str:
    if isinstance(x, int):
        return x
    try:
        x = float(x)
        if x.is_integer():
            return str(int(x))
        return str(x)
    except ValueError:
        return str(x)


# Pyarrow functions declaration for DuckDB
con.create_function("cast_time_types", _cast_time_types)
con.create_function("handle_str_number", _handle_str_number)


class Operator:
    """Superclass for all operators"""

    op: Any = None
    py_op: Any = None
    type_to_check: Any = None
    return_type: Any = None

    @classmethod
    def analyze(cls, *args: Any, **kwargs: Any) -> Any:
        if only_semantic:
            return cls.validate(*args, **kwargs)
        return cls.evaluate(*args, **kwargs)

    @classmethod
    def modify_measure_column(cls, result: Dataset) -> None:
        """
        If an Operator change the data type of the Variable it is applied to (e.g., from string to
        number), the result Data Set cannot maintain this Variable as it happens in the previous
        cases, because a Variable cannot have different data types in different Data Sets.
        As a consequence, the converted variable cannot follow the same rules described in the
        sections above and must be replaced, in the result Data Set, by another Variable of the
        proper data type.
        For sake of simplicity, the operators changing the data type are allowed only on
        mono-measure operand Data Sets, so that the conversion happens on just one Measure.
        A default generic Measure is assigned by default to the result Data Set, depending on the
        data type of the result (the default Measure Variables are reported in the table below).

        Function used by the evaluate function when a dataset is involved
        """

        if len(result.get_measures()) == 1 and cls.return_type is not None and result is not None:
            measure_name = result.get_measures_names()[0]
            components = list(result.components.keys())
            columns = list(result.data.columns) if result.data is not None else []

            transformations = []
            column_to_rename = None
            for column in columns:
                if column not in set(components):
                    column_to_rename = column
                else:
                    transformations.append(f'"{column}"')
            if column_to_rename:
                transformations.append(f'"{column_to_rename}" AS "{measure_name}"')
            result.data = result.data.project(", ".join(transformations))

    @classmethod
    def validate_dataset_type(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate_component_type(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate_scalar_type(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate(cls, *args: Any, **kwargs: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def evaluate(cls, *args: Any, **kwargs: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def scalar_validation(cls, *args: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def component_validation(cls, *args: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate_type_compatibility(cls, *args: Any) -> bool:
        if len(args) == 1:
            operand = args[0]
            return check_unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)
        if len(args) == 2:
            left, right = args
            return check_binary_implicit_promotion(left, right, cls.type_to_check, cls.return_type)
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def type_validation(cls, *args: Any) -> Any:
        if len(args) == 1:
            operand = args[0]
            return unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)
        if len(args) == 2:
            left, right = args
            return binary_implicit_promotion(left, right, cls.type_to_check, cls.return_type)
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def apply_return_type_dataset(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def apply_return_type(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")


def _id_type_promotion_join_keys(
    c_left: Component,
    c_right: Component,
    join_key: str,
    left_data: Optional[DuckDBPyRelation] = None,
    right_data: Optional[DuckDBPyRelation] = None,
) -> tuple[Optional[DuckDBPyRelation], Optional[DuckDBPyRelation]]:
    if not left_data or not right_data:
        return left_data, right_data

    left_type_name = c_left.data_type.__name__
    right_type_name = c_right.data_type.__name__

    if left_type_name == right_type_name:
        return left_data, right_data

    if {left_type_name, right_type_name} <= {"Integer", "Number"}:
        cast_expr = f"CAST({join_key} AS DOUBLE)::INT AS {join_key}, * EXCLUDE({join_key})"
        left_data = left_data.project(cast_expr)
        right_data = right_data.project(cast_expr)
    elif {"String", "Integer", "Number"} & {left_type_name, right_type_name}:
        transformations = [
            f"handle_str_number({join_key}) AS {join_key}" if col == join_key else col
            for col in left_data.columns
        ]
        left_data = left_data.project(", ".join(transformations))

        transformations = [
            f"handle_str_number({join_key}) AS {join_key}" if col == join_key else col
            for col in right_data.columns
        ]
        right_data = right_data.project(", ".join(transformations))

    return left_data, right_data


class Binary(Operator):
    @classmethod
    def op_func(cls, *args: Any) -> Any:
        x, y = args

        if pd.isnull(x) or pd.isnull(y):
            return None
        return cls.py_op(x, y)

    @classmethod
    def apply_operation_series_scalar(
        cls,
        series: Any,
        scalar: Scalar,
        series_left: bool,
    ) -> Any:
        if scalar is None:
            return pd.Series(None, index=series.index)
        if series_left:
            return series.map(lambda x: cls.py_op(x, scalar), na_action="ignore")
        else:
            return series.map(lambda x: cls.py_op(scalar, x), na_action="ignore")

    @classmethod
    def validate(cls, *args: Any) -> Any:
        """
        The main function for validate, applies the implicit promotion (or check it), and
        can do a semantic check too.
        Returns an operand.
        """
        left_operand, right_operand = args

        if isinstance(left_operand, Dataset) and isinstance(right_operand, Dataset):
            return cls.dataset_validation(left_operand, right_operand)
        if isinstance(left_operand, Dataset) and isinstance(right_operand, Scalar):
            return cls.dataset_scalar_validation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Dataset):
            return cls.dataset_scalar_validation(right_operand, left_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Scalar):
            return cls.scalar_validation(left_operand, right_operand)
        if isinstance(left_operand, DataComponent) and isinstance(right_operand, DataComponent):
            return cls.component_validation(left_operand, right_operand)
        if isinstance(left_operand, DataComponent) and isinstance(right_operand, Scalar):
            return cls.component_scalar_validation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, DataComponent):
            return cls.component_scalar_validation(right_operand, left_operand)
        # In operator
        if isinstance(left_operand, Dataset) and isinstance(right_operand, ScalarSet):
            return cls.dataset_set_validation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, ScalarSet):
            return cls.scalar_set_validation(left_operand, right_operand)
        if isinstance(left_operand, DataComponent) and isinstance(right_operand, ScalarSet):
            return cls.component_set_validation(left_operand, right_operand)

    @classmethod
    def dataset_validation(cls, left_operand: Dataset, right_operand: Dataset) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        left_identifiers = left_operand.get_identifiers_names()
        right_identifiers = right_operand.get_identifiers_names()

        use_right_components = len(left_identifiers) < len(right_identifiers)

        left_measures = sorted(left_operand.get_measures(), key=lambda x: x.name)
        right_measures = sorted(right_operand.get_measures(), key=lambda x: x.name)
        left_measures_names = [measure.name for measure in left_measures]
        right_measures_names = [measure.name for measure in right_measures]

        if left_measures_names != right_measures_names:
            raise SemanticError(
                "1-1-14-1",
                op=cls.op,
                left=left_measures_names,
                right=right_measures_names,
            )
        elif len(left_measures) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=left_operand.name)
        for left_measure, right_measure in zip(left_measures, right_measures):
            cls.type_validation(left_measure.data_type, right_measure.data_type)

        # We do not need anymore these variables
        del left_measures
        del right_measures
        del left_measures_names
        del right_measures_names

        join_keys = list(set(left_identifiers).intersection(right_identifiers))
        if len(join_keys) == 0:
            raise SemanticError("1-3-27", op=cls.op)

        # Deleting extra identifiers that we do not need anymore

        base_operand = right_operand if use_right_components else left_operand
        result_components = {
            component_name: copy(component)
            for component_name, component in base_operand.components.items()
            if component.role in [Role.IDENTIFIER, Role.MEASURE]
        }

        for comp in [x for x in result_components.values() if x.role == Role.MEASURE]:
            if comp.name in left_operand.components and comp.name in right_operand.components:
                left_comp = left_operand.components[comp.name]
                right_comp = right_operand.components[comp.name]
                comp.nullable = left_comp.nullable or right_comp.nullable

        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, left_operand, right_operand)
        return result_dataset

    @classmethod
    def dataset_scalar_validation(cls, dataset: Dataset, scalar: Scalar) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if len(dataset.get_measures()) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=dataset.name)

        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in dataset.components.items()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE]
        }
        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, dataset, scalar)
        return result_dataset

    @classmethod
    def scalar_validation(cls, left_operand: Scalar, right_operand: Scalar) -> Scalar:
        if not cls.validate_type_compatibility(left_operand.data_type, right_operand.data_type):
            raise SemanticError(
                "1-1-1-2",
                type_1=left_operand.data_type,
                type_2=right_operand.data_type,
                type_check=cls.type_to_check,
            )
        return Scalar(
            name="result",
            data_type=cls.type_validation(left_operand.data_type, right_operand.data_type),
            value=None,
        )

    @classmethod
    def component_validation(
        cls, left_operand: DataComponent, right_operand: DataComponent
    ) -> DataComponent:
        """
        Validates the compatibility between the types of the components and the operator
        :param left_operand: The left component
        :param right_operand: The right component
        :return: The result data type of the validation
        """
        comp_name = VirtualCounter._new_dc_name()
        result_data_type = cls.type_validation(left_operand.data_type, right_operand.data_type)
        result = DataComponent(
            name=comp_name,
            data_type=result_data_type,
            data=None,
            role=left_operand.role,
            nullable=(left_operand.nullable or right_operand.nullable),
        )

        return result

    @classmethod
    def component_scalar_validation(cls, component: DataComponent, scalar: Scalar) -> DataComponent:
        cls.type_validation(component.data_type, scalar.data_type)
        result = DataComponent(
            name=component.name,
            data_type=cls.type_validation(component.data_type, scalar.data_type),
            data=None,
            role=component.role,
            nullable=component.nullable or scalar is None,
        )
        return result

    @classmethod
    def dataset_set_validation(cls, dataset: Dataset, scalar_set: ScalarSet) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if len(dataset.get_measures()) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=dataset.name)
        for measure in dataset.get_measures():
            cls.type_validation(measure.data_type, scalar_set.data_type)
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in dataset.components.items()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE]
        }

        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, dataset, scalar_set)
        return result_dataset

    @classmethod
    def component_set_validation(
        cls, component: DataComponent, scalar_set: ScalarSet
    ) -> DataComponent:
        comp_name = VirtualCounter._new_dc_name()
        cls.type_validation(component.data_type, scalar_set.data_type)
        result = DataComponent(
            name=comp_name,
            data_type=cls.type_validation(component.data_type, scalar_set.data_type),
            data=None,
            role=Role.MEASURE,
            nullable=component.nullable,
        )
        return result

    @classmethod
    def scalar_set_validation(cls, scalar: Scalar, scalar_set: ScalarSet) -> Scalar:
        cls.type_validation(scalar.data_type, scalar_set.data_type)
        return Scalar(
            name="result",
            data_type=cls.type_validation(scalar.data_type, scalar_set.data_type),
            value=None,
        )

    # The following class method implements the type promotion
    @classmethod
    def type_validation(cls, left_type: Any, right_type: Any) -> Any:
        """
        Validates the compatibility between the types of the operands and the operator
        and give us the result ScalarType of the promotion
        (implicit type promotion : binary_implicit_type_promotion)

        :param left_type: The left operand data type
        :param right_type: The right operand data type

        :return: result ScalarType or exception
        """

        return binary_implicit_promotion(left_type, right_type, cls.type_to_check, cls.return_type)

    # The following class method checks the type promotion
    @classmethod
    def validate_type_compatibility(cls, left: Any, right: Any) -> bool:
        """
        Validates the compatibility between the types of the operands and the operator
        (implicit type promotion : check_binary_implicit_type_promotion)

        :param left: The left operand
        :param right: The right operand

        :return: True if the types are compatible, False otherwise
        """

        return check_binary_implicit_promotion(left, right, cls.type_to_check, cls.return_type)

    @classmethod
    def apply_return_type_dataset(
        cls, result_dataset: Dataset, left_operand: Any, right_operand: Any
    ) -> None:
        """
        Used in dataset's validation.
        Changes the result dataset and give us his final form
        (#TODO: write this explanation in a better way)
        """

        changed_allowed = cls.op in MONOMEASURE_CHANGED_ALLOWED
        is_mono_measure = len(result_dataset.get_measures()) == 1
        for measure in result_dataset.get_measures():
            left_type = left_operand.get_component(measure.name).data_type
            if isinstance(right_operand, (ScalarSet, Scalar)):
                right_type = right_operand.data_type
            else:
                right_type = right_operand.get_component(measure.name).data_type

            result_data_type = cls.type_validation(left_type, right_type)
            if is_mono_measure and left_type.promotion_changed_type(result_data_type):
                component = Component(
                    name=COMP_NAME_MAPPING[result_data_type],
                    data_type=result_data_type,
                    role=Role.MEASURE,
                    nullable=measure.nullable,
                )
                result_dataset.delete_component(measure.name)
                result_dataset.add_component(component)
                if result_dataset.data is not None:
                    result_dataset.data.rename(columns={measure.name: component.name}, inplace=True)
            elif (
                changed_allowed is False
                and is_mono_measure is False
                and left_type.promotion_changed_type(result_data_type)
            ):
                raise SemanticError("1-1-1-4", op=cls.op)
            else:
                measure.data_type = result_data_type

    @classmethod
    def dataset_evaluation(cls, left_operand: Dataset, right_operand: Dataset) -> Dataset:
        result_dataset = cls.dataset_validation(left_operand, right_operand)

        use_right_as_base = False
        if len(left_operand.get_identifiers_names()) < len(right_operand.get_identifiers_names()):
            use_right_as_base = True
            base_operand_data = right_operand.data
            other_operand_data = left_operand.data
        else:
            base_operand_data = left_operand.data
            other_operand_data = right_operand.data

        join_keys = list(
            set(left_operand.get_identifiers_names()).intersection(
                right_operand.get_identifiers_names()
            )
        )

        for join_key in join_keys:
            base_operand_data, other_operand_data = _id_type_promotion_join_keys(
                left_operand.get_component(join_key),
                right_operand.get_component(join_key),
                join_key,
                base_operand_data,
                other_operand_data,
            )

        try:
            # Merge the data
            if base_operand_data is None or other_operand_data is None:
                # TODO: Check if this is the right way to handle empty data and if its lazy
                result_data = duckdb.from_df(pd.DataFrame())
            else:
                result_data = duckdb_merge(
                    base_operand_data,
                    other_operand_data,
                    join_keys,
                    how="inner",
                )
        except Exception as e:
            raise Exception(f"Error merging datasets on Binary Operator: {str(e)}")

        # Measures are the same, using left operand measures names
        transformations = [f"{d}" for d in result_dataset.get_identifiers_names()]
        for me in left_operand.get_measures():
            if cls.op in BINARY_COMPARISON_OPERATORS and me.data_type.__name__ in TIME_TYPES:
                transformations.append(f"""
                    cast_time_types('{me.data_type.__name__}', {me.name}_x) AS {me.name}_x,
                    cast_time_types('{me.data_type.__name__}', {me.name}_y) AS {me.name}_y
                """)
            left, right = (
                (f"{me.name}_y", f"{me.name}_x")
                if use_right_as_base
                else (f"{me.name}_x", f"{me.name}_y")
            )
            transformations.append(apply_bin_op(cls.op, me.name, left, right))

        final_query = f"{', '.join(transformations)}"
        result_data = result_data.project(final_query)

        # Delete attributes from the result data
        attributes = list(
            set(left_operand.get_attributes_names()).union(right_operand.get_attributes_names())
        )
        for att in attributes:
            if att in result_data.columns:
                result_data = result_data.drop(att, axis=1)
            if att + "_x" in result_data.columns:
                result_data = result_data.drop(att + "_x", axis=1)
            if att + "_y" in result_data.columns:
                result_data = result_data.drop(att + "_y", axis=1)

        result_dataset.data = result_data
        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def scalar_evaluation(cls, left_operand: Scalar, right_operand: Scalar) -> Scalar:
        result_scalar = cls.scalar_validation(left_operand, right_operand)
        result_scalar.value = cls.op_func(left_operand.value, right_operand.value)
        return result_scalar

    @classmethod
    def dataset_scalar_evaluation(
        cls, dataset: Dataset, scalar: Scalar, dataset_left: bool = True
    ) -> Dataset:
        result_dataset = cls.dataset_scalar_validation(dataset, scalar)

        if dataset.data is None:
            result_dataset.data = duckdb.from_df(pd.DataFrame())
            return result_dataset

        result_data = dataset.data
        scalar_value = cast_time_types_scalar(cls.op, scalar.data_type, scalar.value)

        transformations = [f"{d}" for d in result_dataset.get_identifiers_names()]
        for me in dataset.get_measures():
            if me.data_type.__name__ in TIME_TYPES:
                transformations.append(
                    f'cast_time_types("{me.data_type.__name__}", {me.name}) AS "{me.name}"'
                )
            if me.data_type.__name__.__str__() == "Duration" and not isinstance(scalar_value, int):
                scalar_value = PERIOD_IND_MAPPING[scalar_value]

            left, right = (me.name, scalar_value) if dataset_left else (scalar_value, me.name)
            transformations.append(apply_bin_op(cls.op, me.name, left, right))

        final_query = f"{', '.join(transformations)}"
        result_dataset.data = result_data.project(final_query)
        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def component_evaluation(
        cls, left_operand: DataComponent, right_operand: DataComponent
    ) -> DataComponent:
        result_component = cls.component_validation(left_operand, right_operand)
        if left_operand.data is None or right_operand.data is None:
            return duckdb.from_df(pd.Series())

        result_data = duckdb_concat(left_operand.data, right_operand.data)

        transformations = ["*"]
        if left_operand.data_type in TIME_TYPES:
            transformations.append(
                f'cast_time_types("{left_operand.data_type.__name__}", {left_operand.name}) '
                f'AS "{left_operand.name}"'
            )
        if right_operand.data_type in TIME_TYPES:
            transformations.append(
                f'cast_time_types("{right_operand.data_type.__name__}", {right_operand.name}) '
                f'AS "{right_operand.name}"'
            )

        transformations.append(
            apply_bin_op(cls.op, result_component.name, left_operand.name, right_operand.name)
        )
        final_query = f"{', '.join(transformations)}"
        result_data = result_data.project(final_query)
        result_component.data = result_data.project(result_component.name)
        return result_component

    @classmethod
    def component_scalar_evaluation(
        cls, component: DataComponent, scalar: Scalar, component_left: bool = True
    ) -> DataComponent:
        result_component = cls.component_scalar_validation(component, scalar)
        comp_data = component.data or duckdb.from_df(pd.Series())

        transformations = []
        if component.data_type.__name__ in TIME_TYPES:
            transformations.append(
                f'cast_time_types("{component.data_type.__name__}", {component.name}) '
                f'AS "{component.name}"'
            )

        scalar_value = cast_time_types_scalar(cls.op, scalar.data_type, scalar.value)
        if component.data_type.__name__.__str__() == "Duration" and not isinstance(
            scalar_value, int
        ):
            scalar_value = PERIOD_IND_MAPPING[scalar_value]

        transformations.append(
            apply_bin_op(cls.op, result_component.name, component.name, scalar_value)
        )
        final_query = f"{', '.join(transformations)}"
        result_component.data = comp_data.project(final_query)
        return result_component

    @classmethod
    def dataset_set_evaluation(cls, dataset: Dataset, scalar_set: ScalarSet) -> Dataset:
        result_dataset = cls.dataset_set_validation(dataset, scalar_set)
        result_data = dataset.data or duckdb.from_df(pd.DataFrame())
        scalar_set.values = (
            scalar_set.values
            if isinstance(scalar_set.values, DuckDBPyRelation)
            else (con.from_arrow_table(duckdb.arrow(scalar_set.values)))
        )

        transformations = [f'"{d}"' for d in dataset.get_identifiers_names()]
        for measure_name in dataset.get_measures_names():
            transformations.append(
                apply_bin_op(cls.op, measure_name, measure_name, scalar_set.values.columns[0])
            )

        result_dataset.data = result_data.project(", ".join(transformations))
        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def component_set_evaluation(
        cls, component: DataComponent, scalar_set: ScalarSet
    ) -> DataComponent:
        result_component = cls.component_set_validation(component, scalar_set)
        result_data = component.data or duckdb.from_df(pd.Series())
        scalar_set.values = (
            scalar_set.values
            if isinstance(scalar_set.values, DuckDBPyRelation)
            else (con.from_arrow_table(duckdb.arrow(scalar_set.values)))
        )

        result_component.data = result_data.project(
            apply_bin_op(
                cls.op, result_component.name, component.name, scalar_set.values.columns[0]
            )
        )
        return result_component

    @classmethod
    def scalar_set_evaluation(cls, scalar: Scalar, scalar_set: ScalarSet) -> Scalar:
        result_scalar = cls.scalar_set_validation(scalar, scalar_set)
        result_scalar.value = cls.op_func(scalar.value, scalar_set)
        return result_scalar

    @classmethod
    def evaluate(cls, left_operand: Any, right_operand: Any) -> Any:
        """
        Evaluate the operation (based on validation output)
        :param left_operand: The left operand
        :param right_operand: The right operand
        :return: The result of the operation
        """

        if isinstance(left_operand, Dataset) and isinstance(right_operand, Dataset):
            return cls.dataset_evaluation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Scalar):
            return cls.scalar_evaluation(left_operand, right_operand)
        if isinstance(left_operand, Dataset) and isinstance(right_operand, Scalar):
            return cls.dataset_scalar_evaluation(left_operand, right_operand, dataset_left=True)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Dataset):
            return cls.dataset_scalar_evaluation(right_operand, left_operand, dataset_left=False)
        if isinstance(left_operand, DataComponent) and isinstance(right_operand, DataComponent):
            return cls.component_evaluation(left_operand, right_operand)
        if isinstance(left_operand, DataComponent) and isinstance(right_operand, Scalar):
            return cls.component_scalar_evaluation(left_operand, right_operand, component_left=True)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, DataComponent):
            return cls.component_scalar_evaluation(
                right_operand, left_operand, component_left=False
            )
        if isinstance(left_operand, Dataset) and isinstance(right_operand, ScalarSet):
            return cls.dataset_set_evaluation(left_operand, right_operand)
        if isinstance(left_operand, DataComponent) and isinstance(right_operand, ScalarSet):
            return cls.component_set_evaluation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, ScalarSet):
            return cls.scalar_set_evaluation(left_operand, right_operand)


class Unary(Operator):
    @classmethod
    def op_func(cls, *args: Any) -> Any:
        x = args[0]

        return None if pd.isnull(x) else cls.py_op(x)

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        """
        Applies the operation to a component
        """

        return series.map(cls.py_op, na_action="ignore")

    @classmethod
    def validate(cls, operand: Any) -> Any:
        """
        The main function for validate, applies the implicit promotion (or check it), and
        can do a semantic check too.
        Returns an operand.
        """

        if isinstance(operand, Dataset):
            return cls.dataset_validation(operand)
        elif isinstance(operand, DataComponent):
            return cls.component_validation(operand)
        elif isinstance(operand, Scalar):
            return cls.scalar_validation(operand)

    @classmethod
    def dataset_validation(cls, operand: Dataset) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        cls.validate_dataset_type(operand)
        if len(operand.get_measures()) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in operand.components.items()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE]
        }

        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, operand)
        return result_dataset

    @classmethod
    def scalar_validation(cls, operand: Scalar) -> Scalar:
        result_type = cls.type_validation(operand.data_type)
        result = Scalar(name="result", data_type=result_type, value=None)
        return result

    @classmethod
    def component_validation(cls, operand: DataComponent) -> DataComponent:
        comp_name = VirtualCounter._new_dc_name()
        result_type = cls.type_validation(operand.data_type)
        result = DataComponent(
            name=comp_name,
            data_type=result_type,
            data=None,
            role=operand.role,
            nullable=operand.nullable,
        )
        return result

    # The following class method implements the type promotion
    @classmethod
    def type_validation(cls, operand: Any) -> Any:
        return unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)

    # The following class method checks the type promotion
    @classmethod
    def validate_type_compatibility(cls, operand: Any) -> bool:
        return check_unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)

    @classmethod
    def validate_dataset_type(cls, dataset: Dataset) -> None:
        if cls.type_to_check is not None:
            for measure in dataset.get_measures():
                if not cls.validate_type_compatibility(measure.data_type):
                    raise SemanticError(
                        "1-1-1-3",
                        op=cls.op,
                        entity=measure.role.value,
                        name=measure.name,
                        target_type=SCALAR_TYPES_CLASS_REVERSE[cls.type_to_check],
                    )

    @classmethod
    def validate_scalar_type(cls, scalar: Scalar) -> None:
        if cls.type_to_check is not None and not cls.validate_type_compatibility(scalar.data_type):
            raise SemanticError(
                "1-1-1-5",
                op=cls.op,
                name=scalar.name,
                type=SCALAR_TYPES_CLASS_REVERSE[scalar.data_type],
            )

    @classmethod
    def apply_return_type_dataset(cls, result_dataset: Dataset, operand: Dataset) -> None:
        changed_allowed = cls.op in MONOMEASURE_CHANGED_ALLOWED
        is_mono_measure = len(operand.get_measures()) == 1
        for measure in result_dataset.get_measures():
            operand_type = operand.get_component(measure.name).data_type

            result_data_type = cls.type_validation(operand_type)
            if is_mono_measure and operand_type.promotion_changed_type(result_data_type):
                component = Component(
                    name=COMP_NAME_MAPPING[result_data_type],
                    data_type=result_data_type,
                    role=Role.MEASURE,
                    nullable=measure.nullable,
                )
                result_dataset.delete_component(measure.name)
                result_dataset.add_component(component)
                if result_dataset.data is not None:
                    result_dataset.data.rename(columns={measure.name: component.name}, inplace=True)
            elif (
                changed_allowed is False
                and is_mono_measure is False
                and operand_type.promotion_changed_type(result_data_type)
            ):
                raise SemanticError("1-1-1-4", op=cls.op)
            else:
                measure.data_type = result_data_type

    @classmethod
    def evaluate(cls, operand: ALL_MODEL_DATA_TYPES) -> Any:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand)

    @classmethod
    def dataset_evaluation(cls, operand: Dataset) -> Dataset:
        result_dataset = cls.dataset_validation(operand)
        result_data = operand.data or duckdb.from_df(pd.DataFrame())

        transformations = [f'"{d}"' for d in operand.get_identifiers_names()]
        for measure_name in operand.get_measures_names():
            transformations.append(apply_unary_op(cls.op, measure_name, measure_name))

        result_dataset.data = result_data.project(", ".join(transformations))
        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def scalar_evaluation(cls, operand: Scalar) -> Scalar:
        result_scalar = cls.scalar_validation(operand)
        result_scalar.value = cls.op_func(operand.value)
        return result_scalar

    @classmethod
    def component_evaluation(cls, operand: DataComponent) -> DataComponent:
        result_component = cls.component_validation(operand)
        result_data = operand.data or duckdb.from_df(pd.Series())
        result_component.data = result_data.project(
            apply_unary_op(cls.op, operand.name, result_component.name)
        )
        return result_component
