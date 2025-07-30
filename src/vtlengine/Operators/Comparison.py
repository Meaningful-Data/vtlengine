import operator
import re
import uuid
from copy import copy
from typing import Any, Optional, Union

import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import (
    CHARSET_MATCH,
    EQ,
    GT,
    GTE,
    IN,
    ISNULL,
    LT,
    LTE,
    NEQ,
    NOT_IN,
)
from vtlengine.connection import con
from vtlengine.DataTypes import COMP_NAME_MAPPING, Boolean, Null, Number, String
from vtlengine.duckdb.custom_functions import between_duck, isnull_duck
from vtlengine.duckdb.duckdb_utils import duckdb_concat, empty_relation
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar, ScalarSet
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


class Unary(Operator.Unary):
    """
    Unary comparison operator. It returns a boolean.
    """

    return_type = Boolean


class IsNull(Unary):
    """
    Class that allows to perform the isnull comparison operator.
    It has different class methods to allow performing the operation with different datatypes.
    """

    op = ISNULL
    py_op = isnull_duck
    sql_op = "isnull_duck"

    @classmethod
    def dataset_validation(cls, operand: Dataset) -> Dataset:
        result = super().dataset_validation(operand)
        for measure in result.get_measures():
            measure.nullable = False
        return result

    @classmethod
    def component_validation(cls, operand: DataComponent) -> DataComponent:
        result = super().component_validation(operand)
        result.nullable = False
        return result


class Binary(Operator.Binary):
    """
    Binary comparison operator. It returns a boolean.
    """

    return_type = Boolean

    @classmethod
    def _cast_values(
        cls,
        x: Optional[Union[int, float, str, bool]],
        y: Optional[Union[int, float, str, bool]],
    ) -> Any:
        # Cast values to compatible types for comparison
        try:
            if isinstance(x, str) and isinstance(y, bool):
                y = String.cast(y)
            elif isinstance(x, bool) and isinstance(y, str):
                x = String.cast(x)
            elif isinstance(x, str) and isinstance(y, (int, float)):
                x = Number.cast(x)
            elif isinstance(x, (int, float)) and isinstance(y, str):
                y = Number.cast(y)
        except ValueError:
            x = str(x)
            y = str(y)

        return x, y

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        # Return None if any of the values are NaN
        if pd.isnull(x) or pd.isnull(y):
            return None
        x, y = cls._cast_values(x, y)
        return cls.py_op(x, y)

    @classmethod
    def apply_operation_series_scalar(cls, series: Any, scalar: Any, series_left: bool) -> Any:
        if pd.isnull(scalar):
            return pd.Series(None, index=series.index)

        first_non_null = series.dropna().iloc[0] if not series.dropna().empty else None
        if first_non_null is not None:
            scalar, first_non_null = cls._cast_values(scalar, first_non_null)

            series_type = pd.api.types.infer_dtype(series, skipna=True)
            first_non_null_type = pd.api.types.infer_dtype([first_non_null])

            if series_type != first_non_null_type:
                if isinstance(first_non_null, str):
                    series = series.astype(str)
                elif isinstance(first_non_null, (int, float)):
                    series = series.astype(float)

        op = cls.py_op if cls.py_op is not None else cls.op_func
        if series_left:
            result = series.map(lambda x: op(x, scalar), na_action="ignore")
        else:
            result = series.map(lambda x: op(scalar, x), na_action="ignore")

        return result

    @classmethod
    def apply_return_type_dataset(
        cls,
        result_dataset: Dataset,
        left_operand: Dataset,
        right_operand: Union[Dataset, Scalar, ScalarSet],
    ) -> None:
        super().apply_return_type_dataset(result_dataset, left_operand, right_operand)
        is_mono_measure = len(result_dataset.get_measures()) == 1
        if is_mono_measure:
            measure = result_dataset.get_measures()[0]
            component = Component(
                name=COMP_NAME_MAPPING[Boolean],
                data_type=Boolean,
                role=Role.MEASURE,
                nullable=measure.nullable,
            )
            result_dataset.delete_component(measure.name)
            result_dataset.add_component(component)
            if result_dataset.data is not None:
                result_dataset.data.rename(columns={measure.name: component.name}, inplace=True)


class Equal(Binary):
    op = EQ
    py_op = operator.eq


class NotEqual(Binary):
    op = NEQ
    py_op = operator.ne


class Greater(Binary):
    op = GT
    py_op = operator.gt


class GreaterEqual(Binary):
    op = GTE
    py_op = operator.ge


class Less(Binary):
    op = LT
    py_op = operator.lt


class LessEqual(Binary):
    op = LTE
    py_op = operator.le


class In(Binary):
    op = IN

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: ScalarSet) -> Any:
        if right_series.data_type == Null:
            return pd.Series(None, index=left_series.index)

        return left_series.map(lambda x: x in right_series, na_action="ignore")

    @classmethod
    def py_op(cls, x: Any, y: Any) -> Any:
        if y.data_type == Null:
            return None
        return operator.contains(y, x)


class NotIn(Binary):
    op = NOT_IN

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: Any) -> Any:
        series_result = In.apply_operation_two_series(left_series, right_series)
        return series_result.map(lambda x: not x, na_action="ignore")

    @classmethod
    def py_op(cls, x: Any, y: Any) -> Any:
        return not operator.contains(y, x)


class Match(Binary):
    op = CHARSET_MATCH
    type_to_check = String

    @classmethod
    def op_func(cls, x: Optional[str], y: Optional[str]) -> Optional[bool]:
        if pd.isnull(x) or pd.isnull(y):
            return None
        if isinstance(x, pd.Series):
            return x.str.fullmatch(y)
        return bool(re.fullmatch(str(y), str(x)))


class Between(Operator.Operator):
    return_type = Boolean
    sql_op = "between_duck"
    py_op = between_duck
    """
    This comparison operator has the following class methods.

    Class methods:
        op_function: Sets the data to be manipulated.
        apply_operation_component: Returns a pandas dataframe with the operation,

        considering each component with the schema of op_function.

        apply_return_type_dataset: Because the result must be a boolean,
        this function evaluates if the measure is actually a boolean one.
    """

    @classmethod
    def apply_between_op(cls, input_column_name: str, from_: Any, to: Any) -> str:
        return f"{cls.sql_op}({input_column_name}, {from_}, {to})"

    @classmethod
    def apply_return_type_dataset(cls, result_dataset: Dataset, operand: Dataset) -> None:
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
            elif is_mono_measure is False and operand_type.promotion_changed_type(result_data_type):
                raise SemanticError("1-1-1-4", op=cls.op)
            else:
                measure.data_type = result_data_type

    @classmethod
    def validate(
        cls,
        operand: Union[Dataset, DataComponent, Scalar],
        from_: Union[DataComponent, Scalar],
        to: Union[DataComponent, Scalar],
    ) -> Any:
        result: Union[Dataset, DataComponent, Scalar]
        if isinstance(operand, Dataset):
            if len(operand.get_measures()) == 0:
                raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)
            result_components = {
                comp_name: copy(comp)
                for comp_name, comp in operand.components.items()
                if comp.role == Role.IDENTIFIER or comp.role == Role.MEASURE
            }
            result = Dataset(name=operand.name, components=result_components, data=None)
        elif isinstance(operand, DataComponent):
            result = DataComponent(
                name=operand.name,
                data=None,
                data_type=cls.return_type,
                role=operand.role,
            )
        elif isinstance(from_, Scalar) and isinstance(to, Scalar):
            result = Scalar(name=operand.name, value=None, data_type=cls.return_type)
        else:  # From or To is a DataComponent, or both
            result = DataComponent(
                name=operand.name,
                data=None,
                data_type=cls.return_type,
                role=Role.MEASURE,
            )

        if isinstance(operand, Dataset):
            for measure in operand.get_measures():
                cls.validate_type_compatibility(measure.data_type, from_.data_type)
                cls.validate_type_compatibility(measure.data_type, to.data_type)
                if isinstance(result, Dataset):
                    cls.apply_return_type_dataset(result, operand)
        else:
            cls.validate_type_compatibility(operand.data_type, from_.data_type)
            cls.validate_type_compatibility(operand.data_type, to.data_type)

        return result

    @staticmethod
    def handle_param_value(param: Optional[Union[DataComponent, Scalar]]) -> Any:
        if isinstance(param, DataComponent):
            return f'"{param.name}"'
        elif isinstance(param, Scalar):
            return "NULL" if param.value is None else param.value
        return "NULL"

    @classmethod
    def evaluate(
        cls,
        operand: Operator.ALL_MODEL_DATA_TYPES,
        from_: Union[DataComponent, Scalar],
        to: Union[DataComponent, Scalar],
    ) -> Any:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, from_, to)
        if (
            isinstance(operand, DataComponent)
            or isinstance(from_, DataComponent)
            or isinstance(to, DataComponent)
        ):
            return cls.component_evaluation(operand, from_, to)

        return cls.scalar_evaluation(operand, from_, to)

    @classmethod
    def dataset_evaluation(cls, *args: Any) -> Dataset:
        operand, from_, to = args[:3]

        result_dataset = cls.validate(operand, from_, to)
        result_data = operand.data if operand.data is not None else empty_relation()

        expr = [f"{d}" for d in operand.get_identifiers_names()]

        from_value = cls.handle_param_value(from_)
        to_value = cls.handle_param_value(to)

        for measure_name in operand.get_measures_names():
            expr.append(cls.apply_between_op(measure_name, from_value, to_value))

        result_dataset.data = result_data.project(", ".join(expr))
        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def scalar_evaluation(cls, *args: Scalar) -> Scalar:
        operand, from_, to = args[:3]

        result = cls.validate(operand, from_, to)

        from_value = cls.handle_param_value(from_)
        to_value = cls.handle_param_value(to)
        result.value = cls.py_op(operand.value, from_value, to_value)
        return result

    @classmethod
    def component_evaluation(cls, *args: Union[DataComponent, Scalar]) -> DataComponent:
        operand, from_, to = args[:3]

        result = cls.validate(
            operand,
            from_,
            to,
        )
        operand_value = cls.handle_param_value(operand)
        from_value = cls.handle_param_value(from_)
        to_value = cls.handle_param_value(to)

        # Any component can drive the evaluation,
        # so we need to concat all component data into the same relation
        all_data = None

        for param in args:
            if isinstance(param, DataComponent):
                if all_data is None:
                    all_data = param.data if param.data is not None else empty_relation()
                else:
                    all_data = duckdb_concat(all_data, param.data)

        result.data = all_data.project(  # type: ignore[union-attr]
            cls.apply_between_op(operand_value, from_value, to_value)
        )

        return result


class ExistIn(Operator.Operator):
    """
    Class methods:
        validate: Sets the identifiers and check if the left one exists in the right one.
        evaluate: Evaluates if the result data type is actually a boolean.
    """

    op = IN

    # noinspection PyTypeChecker
    @classmethod
    def validate(
        cls, dataset_1: Dataset, dataset_2: Dataset, retain_element: Optional[bool]
    ) -> Any:
        dataset_name = VirtualCounter._new_ds_name()
        left_identifiers = dataset_1.get_identifiers_names()
        right_identifiers = dataset_2.get_identifiers_names()

        is_subset_right = set(right_identifiers).issubset(left_identifiers)
        is_subset_left = set(left_identifiers).issubset(right_identifiers)
        if not (is_subset_left or is_subset_right):
            raise ValueError("Datasets must have common identifiers")

        result_components = {comp.name: copy(comp) for comp in dataset_1.get_identifiers()}
        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        result_dataset.add_component(
            Component(name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=False)
        )
        return result_dataset

    @classmethod
    def evaluate(
        cls, dataset_1: Dataset, dataset_2: Dataset, retain_element: Optional[bool]
    ) -> Any:
        result_dataset = cls.validate(dataset_1, dataset_2, retain_element)

        id_names = dataset_1.get_identifiers_names()
        op1_name = VirtualCounter._new_temp_view_name()
        op2_name = VirtualCounter._new_temp_view_name()

        con.register(op1_name, dataset_1.data)
        con.register(op2_name, dataset_2.data)

        ids_str = ", ".join(id_names)
        select_str = ", ".join([f"{op1_name}.{col}" for col in id_names])
        case_expr = "CASE WHEN bool_var IS NULL THEN False ELSE True END AS bool_var"

        retain_all_query = f"""
                SELECT {select_str.replace(op1_name, "__vds_exists__")}, {case_expr}
                FROM {op1_name} __vds_exists__
                LEFT JOIN (
                    SELECT * FROM (
                        SELECT *, True AS bool_var FROM {op1_name}
                    ) SEMI JOIN {op2_name} USING ({ids_str})
                )
                USING ({ids_str})
            """

        retain_true_query = f"""
                SELECT * FROM (
                    SELECT {select_str}, True AS bool_var FROM {op1_name}
                ) SEMI JOIN {op2_name} USING ({ids_str})
            """

        retain_false_query = f"""
                SELECT * FROM (
                    SELECT {select_str}, False AS bool_var FROM {op1_name}
                ) ANTI JOIN {op2_name} USING ({ids_str})
            """

        if retain_element is None:
            query = retain_all_query
        elif retain_element is False:
            query = retain_false_query
        else:
            query = retain_true_query

        result_dataset.data = con.query(query)
        return result_dataset

    @staticmethod
    def _check_all_columns(row: Any) -> bool:
        return all(col_value == True for col_value in row)
