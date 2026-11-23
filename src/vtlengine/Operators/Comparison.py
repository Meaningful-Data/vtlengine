import operator
import re
from copy import copy
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import (
    BETWEEN,
    CHARSET_MATCH,
    EQ,
    EXISTS_IN,
    GT,
    GTE,
    IN,
    ISNULL,
    LT,
    LTE,
    NEQ,
    NOT_IN,
)
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    Boolean,
    Null,
    String,
    TimeInterval,
    to_string,
    to_strings,
)
from vtlengine.Exceptions import RunTimeError, SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar, ScalarSet, values_of
from vtlengine.Operators.Boolean import And
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.Utils._dataframe import merge_frames
from vtlengine.Utils._number_config import (
    numbers_are_equal,
    numbers_are_greater_equal,
    numbers_are_less_equal,
)


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
    py_op = pd.isnull

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return series.isnull().astype("bool[pyarrow]")

    @classmethod
    def op_func(cls, x: Any) -> Any:
        return pd.isnull(x)

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

    @classmethod
    def scalar_validation(cls, operand: Scalar) -> Scalar:
        result = super().scalar_validation(operand)
        result.nullable = False
        return result


def _tolerance_applies(*values: Any) -> bool:
    """Number comparisons use the configured tolerance only when a float is
    involved: the tolerance absorbs float representation noise, so two exact
    integers compare exactly — matching the DuckDB engine, whose BIGINT
    comparisons never go through the tolerance macros (issue #985).
    """
    return all(isinstance(v, (int, float)) for v in values) and any(
        isinstance(v, float) for v in values
    )


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
        # A Boolean compared with a String is promoted to String, also a numpy Boolean, which
        # is no bool (issue #1314)
        if isinstance(x, str) and isinstance(y, (bool, np.bool_)):
            y = String.cast(y)
        elif isinstance(x, (bool, np.bool_)) and isinstance(y, str):
            x = String.cast(x)

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
            return pd.Series(None, index=series.index, dtype="bool[pyarrow]")

        first_non_null = series.dropna().iloc[0] if not series.dropna().empty else None
        if first_non_null is not None:
            scalar, first_non_null = cls._cast_values(scalar, first_non_null)

            series_type = pd.api.types.infer_dtype(series, skipna=True)
            first_non_null_type = pd.api.types.infer_dtype([first_non_null])

            if series_type != first_non_null_type:
                if isinstance(first_non_null, str):
                    series = to_strings(series)
                elif isinstance(first_non_null, (int, float)):
                    series = series.astype("double[pyarrow]")

        # Use op_func if it's overridden (not from Binary base class)
        # to support tolerance-based number comparisons
        if cls.op_func is not Binary.op_func:
            op = cls.op_func
        else:
            op = cls.py_op if cls.py_op is not None else cls.op_func

        if series_left:
            result = series.map(lambda x: op(x, scalar), na_action="ignore")
        else:
            result = series.map(lambda x: op(scalar, x), na_action="ignore")

        return result.astype("bool[pyarrow]")

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

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        # Return None if any of the values are NaN
        if pd.isnull(x) or pd.isnull(y):
            return None
        x, y = cls._cast_values(x, y)

        # Use tolerance-based comparison for float-involved numeric types
        if _tolerance_applies(x, y):
            return numbers_are_equal(x, y)

        return cls.py_op(x, y)


class NotEqual(Binary):
    op = NEQ
    py_op = operator.ne

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        # Return None if any of the values are NaN
        if pd.isnull(x) or pd.isnull(y):
            return None
        x, y = cls._cast_values(x, y)

        # Use tolerance-based comparison for float-involved numeric types
        if _tolerance_applies(x, y):
            return not numbers_are_equal(x, y)

        return cls.py_op(x, y)


class Ordering(Binary):
    """
    Comparison operators that order their operands, which Time values do not support.
    """

    @classmethod
    def type_validation(cls, left_type: Any, right_type: Any) -> Any:
        # The order of two Time values depends only on their type, so the validation
        # rejects it and the semantic analysis raises the error (issue #1192)
        if TimeInterval in (left_type, right_type):
            raise SemanticError("1-1-19-18", op=cls.op)
        return super().type_validation(left_type, right_type)


class Greater(Ordering):
    op = GT
    py_op = operator.gt


class GreaterEqual(Ordering):
    op = GTE
    py_op = operator.ge

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        # Return None if any of the values are NaN
        if pd.isnull(x) or pd.isnull(y):
            return None
        x, y = cls._cast_values(x, y)

        # Use tolerance-based comparison for float-involved numeric types
        if _tolerance_applies(x, y):
            return numbers_are_greater_equal(x, y)

        return cls.py_op(x, y)


class Less(Ordering):
    op = LT
    py_op = operator.lt


class LessEqual(Ordering):
    op = LTE
    py_op = operator.le

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        # Return None if any of the values are NaN
        if pd.isnull(x) or pd.isnull(y):
            return None
        x, y = cls._cast_values(x, y)

        # Use tolerance-based comparison for float-involved numeric types
        if _tolerance_applies(x, y):
            return numbers_are_less_equal(x, y)

        return cls.py_op(x, y)


def _time_in_operands(operand: Any, values: Any) -> Any:
    """The operand of in / not_in and its set of values, compared as Time values where a Date
    or a Time_Period meets a Time: the one of them that is a Date or a Time_Period is cast to
    Time (issue #1352). A String operand is compared with a set of Booleans as text: the
    Boolean values are cast to the Strings "true" and "false", as a Boolean is implicitly cast
    to String and a String never to Boolean (issue #1353)."""
    if not isinstance(values, ScalarSet):
        return operand, values
    if isinstance(operand, Dataset):
        types = {comp.data_type for comp in operand.get_measures()}
        if values.data_type is TimeInterval:
            return Operator.to_time(operand, operand.get_measures_names()), values
    else:
        types = {operand.data_type}
        if values.data_type is TimeInterval:
            return Operator.to_time(operand), values
    if values.data_type is Boolean and types == {String}:
        texts: List[Any] = [to_string(v) for v in values.values]
        return operand, ScalarSet(data_type=String, values=texts)
    return operand, (Operator.to_time(values) if types == {TimeInterval} else values)


class In(Binary):
    op = IN

    @classmethod
    def evaluate(cls, left_operand: Any, right_operand: Any) -> Any:
        return super().evaluate(*_time_in_operands(left_operand, right_operand))

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: ScalarSet) -> Any:
        if right_series.data_type == Null:
            return pd.Series(None, index=left_series.index, dtype="bool[pyarrow]")
        if right_series.data_type == String:
            # A Boolean operand is promoted to String (issue #1314)
            left_series = to_strings(left_series)
        # A value is compared with the values of the set as a value of its type, whatever form
        # it is written in (issue #1349)
        left_series = values_of(right_series.data_type, left_series)
        values = right_series.compared_values()
        if pd.api.types.is_float_dtype(left_series):
            # -0.0 is 0.0, as for =, where isin hashes them apart (issue #1353)
            left_series = left_series + 0.0
            values = [value + 0.0 if isinstance(value, float) else value for value in values]
        result = left_series.isin(values)
        return result.where(left_series.notna(), other=pd.NA).astype("bool[pyarrow]")

    @classmethod
    def py_op(cls, x: Any, y: Any) -> Any:
        if y.data_type == Null:
            return None
        return operator.contains(y, x)


class NotIn(Binary):
    op = NOT_IN

    @classmethod
    def evaluate(cls, left_operand: Any, right_operand: Any) -> Any:
        return super().evaluate(*_time_in_operands(left_operand, right_operand))

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: Any) -> Any:
        series_result = In.apply_operation_two_series(left_series, right_series)
        return (~series_result).astype("bool[pyarrow]")

    @classmethod
    def py_op(cls, x: Any, y: Any) -> Any:
        # The negation of in: null for a set of nulls (issue #1353)
        result = In.py_op(x, y)
        return None if result is None else not result


# The POSIX character classes, as RE2 reads them on the DuckDB engine (ASCII)
_POSIX_CLASSES = {
    "alnum": "0-9A-Za-z",
    "alpha": "A-Za-z",
    "ascii": "\\x00-\\x7F",
    "blank": "\\t ",
    "cntrl": "\\x00-\\x1F\\x7F",
    "digit": "0-9",
    "graph": "!-~",
    "lower": "a-z",
    "print": " -~",
    "punct": "!-/:-@\\[-`{-~",
    "space": "\\t\\n\\v\\f\\r ",
    "upper": "A-Z",
    "word": "0-9A-Za-z_",
    "xdigit": "0-9A-Fa-f",
}
_POSIX_CLASS = re.compile(r"\[:([a-z]+):\]")


def posix_classes(pattern: str, strict: bool = False) -> str:
    """The pattern with the POSIX character classes of its bracket expressions, such as
    [[:alpha:]], written as the ranges Python re reads: the pattern is a POSIX Extended
    Regular Expression in the VTL 2.2 manual, read so by RE2 on the DuckDB engine (issue
    #1353). With strict, a class RE2 does not know, such as [:foo:], raises re.error, as
    RE2 does (issue #1356)."""
    out: List[str] = []
    i, in_set = 0, False
    while i < len(pattern):
        char = pattern[i]
        match = _POSIX_CLASS.match(pattern, i) if in_set else None
        if match is not None and match.group(1) in _POSIX_CLASSES:
            out.append(_POSIX_CLASSES[match.group(1)])
            i = match.end()
            continue
        if match is not None and strict:
            raise re.error(f"unknown POSIX character class {match.group(0)}", pattern, i)
        if char == "\\":
            out.append(pattern[i : i + 2])
            i += 2
            continue
        if char == "[" and not in_set:
            # A ] right after the [ or the [^ is a character of the set
            start = i + 2 if pattern[i + 1 : i + 2] == "^" else i + 1
            end = start + 1 if pattern[start : start + 1] == "]" else start
            out.append(pattern[i:end])
            i, in_set = end, True
            continue
        in_set = in_set and char != "]"
        out.append(char)
        i += 1
    return "".join(out)


class Match(Binary):
    op = CHARSET_MATCH
    type_to_check = String

    @classmethod
    def check_pattern(cls, pattern: str) -> None:
        """A pattern that is not a regular expression, read as op_func reads it (issue
        #1356)."""
        try:
            re.compile(posix_classes(pattern, strict=True))
        except re.error as error:
            raise SemanticError(
                "1-1-18-8", op=cls.op, msg=error.msg, regexp=pattern, pos=error.pos
            ) from None

    @classmethod
    def op_func(cls, x: Optional[str], y: Optional[str]) -> Optional[bool]:
        if pd.isnull(x) or pd.isnull(y):
            return None
        pattern = posix_classes(str(y))
        if isinstance(x, pd.Series):
            return x.str.fullmatch(pattern)
        return bool(re.fullmatch(pattern, to_string(x)))


class Between(Operator.Operator):
    op = BETWEEN
    return_type = Boolean
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
    def op_func(
        cls,
        x: Optional[Union[int, float, bool, str]],
        y: Optional[Union[int, float, bool, str]],
        z: Optional[Union[int, float, bool, str]],
    ) -> Optional[bool]:
        """Op >= from and op <= to, as the VTL 2.2 manual defines between: the functions of
        the >= and <= operators on the values as they order them (see _ordered), so periods
        of different indicators raise 2-1-19-19 also where the other bound is null, joined
        by the and of three-valued logic, false where one of them is false, also when the
        other is null (issue #1349)."""
        from_ok = GreaterEqual.op_func(x, y)
        to_ok = LessEqual.op_func(x, z)
        return And.py_op(
            None if from_ok is None else bool(from_ok), None if to_ok is None else bool(to_ok)
        )

    @classmethod
    def apply_operation_component(cls, series: Any, from_data: Any, to_data: Any) -> Any:
        control_any_series_from_to = isinstance(from_data, pd.Series) or isinstance(
            to_data, pd.Series
        )
        if control_any_series_from_to:
            # A scalar bound keeps its value, which may be of another type than the operand
            if not isinstance(from_data, pd.Series):
                from_data = pd.Series(from_data, index=series.index, dtype=object)
            if not isinstance(to_data, pd.Series):
                to_data = pd.Series(to_data, index=series.index, dtype=object)
            df = pd.DataFrame({"operand": series, "from_data": from_data, "to_data": to_data})
            return df.apply(  # type: ignore[call-overload]
                lambda x: cls.op_func(x["operand"], x["from_data"], x["to_data"]),
                axis=1,
            ).astype("bool[pyarrow]")

        return series.map(lambda x: cls.op_func(x, from_data, to_data)).astype("bool[pyarrow]")

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
        from_: Union[Dataset, DataComponent, Scalar],
        to: Union[Dataset, DataComponent, Scalar],
    ) -> Any:
        result: Union[Dataset, DataComponent, Scalar]
        if isinstance(from_, Dataset) or isinstance(to, Dataset):
            # A bound is a Component or a Scalar (issue #1356)
            raise SemanticError("1-1-15-8", op=cls.op, comp_type="Dataset")
        if isinstance(operand, Dataset):
            if len(operand.get_measures()) == 0:
                raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)
            result_components = {
                comp_name: copy(comp)
                for comp_name, comp in operand.components.items()
                if comp.role in [Role.IDENTIFIER, Role.MEASURE, Role.VIRAL_ATTRIBUTE]
            }
            result = Dataset(name=operand.name, components=result_components, data=None)
        elif isinstance(operand, DataComponent):
            result = DataComponent(
                name=operand.name,
                data=None,
                data_type=cls.return_type,
                role=operand.role,
                nullable=operand.nullable,
            )
        elif isinstance(from_, Scalar) and isinstance(to, Scalar):
            nullable = operand.nullable or from_.nullable or to.nullable
            result = Scalar(
                name=operand.name, value=None, data_type=cls.return_type, nullable=nullable
            )
        else:
            # From or To is a DataComponent, or both
            result = DataComponent(
                name=operand.name,
                data=None,
                data_type=cls.return_type,
                role=Role.MEASURE,
            )

        operand_types = (
            [measure.data_type for measure in operand.get_measures()]
            if isinstance(operand, Dataset)
            else [operand.data_type]
        )
        if TimeInterval in (*operand_types, from_.data_type, to.data_type):
            # between orders its operands, which Time values do not support (issue #1192)
            raise SemanticError("1-1-19-18", op=cls.op)

        # The types are checked as >= and <= check them (issue #1349)
        for operand_type in operand_types:
            GreaterEqual.type_validation(operand_type, from_.data_type)
            LessEqual.type_validation(operand_type, to.data_type)
        if isinstance(result, Dataset) and isinstance(operand, Dataset):
            cls.apply_return_type_dataset(result, operand)

        return result

    @staticmethod
    def _ordered(element: Any, data: Any) -> Any:
        """The values of the element as >= and <= order them: a Time_Period as a period and
        a Duration as its rank, not as their text (issue #1349)."""
        if data is None:
            return data
        if isinstance(data, pd.Series):
            return GreaterEqual.cast_time_types(element.data_type, data)
        return (
            None
            if pd.isnull(data)
            else GreaterEqual.cast_time_types_scalar(element.data_type, data)
        )

    @classmethod
    def evaluate(
        cls,
        operand: Union[DataComponent, Scalar],
        from_: Union[DataComponent, Scalar],
        to: Union[DataComponent, Scalar],
    ) -> Any:
        result = cls.validate(operand, from_, to)
        from_data = cls._ordered(
            from_, from_.data if isinstance(from_, DataComponent) else from_.value
        )
        to_data = cls._ordered(to, to.data if isinstance(to, DataComponent) else to.value)

        if (
            isinstance(from_data, pd.Series)
            and isinstance(to_data, pd.Series)
            and len(from_data) != len(to_data)
        ):
            raise RunTimeError("2-1-1-18", op=cls.op)

        if isinstance(operand, Dataset):
            result.data = operand.data.copy()
            for measure_name in operand.get_measures_names():
                values = cls._ordered(operand.components[measure_name], operand.data[measure_name])
                result.data[measure_name] = cls.apply_operation_component(
                    values, from_data, to_data
                )
                if len(result.get_measures()) == 1:
                    result.data[COMP_NAME_MAPPING[cls.return_type]] = result.data[measure_name]
                    result.data = result.data.drop(columns=[measure_name])
            result.data = result.data[result.get_components_names()]
        if isinstance(operand, DataComponent):
            result.data = cls.apply_operation_component(
                cls._ordered(operand, operand.data), from_data, to_data
            )
        if isinstance(operand, Scalar) and isinstance(from_, Scalar) and isinstance(to, Scalar):
            result.value = cls.op_func(cls._ordered(operand, operand.value), from_data, to_data)
        elif isinstance(operand, Scalar) and (
            isinstance(from_data, pd.Series) or isinstance(to_data, pd.Series)
        ):  # From or To is a DataComponent, or both
            # The operand keeps its value, which >= and <= compare as they do in the pair: a
            # Boolean next to a String bound is written as cast to string writes it
            index = from_data.index if isinstance(from_data, pd.Series) else to_data.index
            series = pd.Series(cls._ordered(operand, operand.value), index=index, dtype=object)
            result_series = cls.apply_operation_component(series, from_data, to_data)
            result = DataComponent(
                name=operand.name,
                data=result_series,
                data_type=cls.return_type,
                role=Role.MEASURE,
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
        cls, dataset_1: Dataset, dataset_2: Dataset, retain_element: Optional[Boolean]
    ) -> Any:
        dataset_name = VirtualCounter._new_ds_name()
        left_identifiers = dataset_1.get_identifiers_names()
        right_identifiers = dataset_2.get_identifiers_names()

        left_ids_set = set(left_identifiers)
        right_ids_set = set(right_identifiers)
        if not left_ids_set or not right_ids_set:
            # The Data Points are matched by their Identifiers, as in a join (issue #1356)
            raise SemanticError("1-2-10", op=EXISTS_IN)
        if not (left_ids_set.issubset(right_ids_set) or right_ids_set.issubset(left_ids_set)):
            raise SemanticError(
                "1-2-15",
                op=cls.op,
                left_name=dataset_1.name,
                left=sorted(left_ids_set),
                right_name=dataset_2.name,
                right=sorted(right_ids_set),
            )
        Operator.check_common_identifier_types(cls.op, dataset_1, dataset_2)

        result_components = {comp.name: copy(comp) for comp in dataset_1.get_identifiers()}
        Operator.common_identifiers(result_components, dataset_1, dataset_2)
        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        result_dataset.add_component(
            Component(name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=False)
        )
        return result_dataset

    @classmethod
    def evaluate(
        cls, dataset_1: Dataset, dataset_2: Dataset, retain_element: Optional[Boolean]
    ) -> Any:
        result_dataset = cls.validate(dataset_1, dataset_2, retain_element)
        dataset_1 = Operator.promoted_keys(dataset_1, result_dataset)
        dataset_2 = Operator.promoted_keys(dataset_2, result_dataset)

        # Checking the subset
        left_id_names = dataset_1.get_identifiers_names()
        right_id_names = dataset_2.get_identifiers_names()
        is_subset_left = set(left_id_names).issubset(right_id_names)

        # Identifiers for the result dataset
        reference_identifiers_names = left_id_names

        # Checking if the left dataset is a subset of the right dataset
        common_columns = left_id_names if is_subset_left else right_id_names

        # Check if the common identifiers are equal between the two datasets
        if dataset_1.data is not None and dataset_2.data is not None:
            true_results = merge_frames(
                dataset_1.data,
                dataset_2.data,
                how="inner",
                left_on=common_columns,
                right_on=common_columns,
            )
            true_results = true_results[reference_identifiers_names]
        else:
            true_results = pd.DataFrame(columns=reference_identifiers_names)

        # Check for empty values
        if true_results.empty:
            true_results["bool_var"] = None
        else:
            true_results["bool_var"] = True
        if dataset_1.data is None:
            dataset_1.data = pd.DataFrame(columns=reference_identifiers_names)
        # Only the Identifiers of dataset_1: a bool_var Measure of it is replaced, not merged
        final_result = merge_frames(
            dataset_1.data[reference_identifiers_names],
            true_results,
            how="left",
            left_on=reference_identifiers_names,
            right_on=reference_identifiers_names,
        )
        final_result = final_result[reference_identifiers_names + ["bool_var"]]

        # No null values are returned, only True or False
        final_result["bool_var"] = final_result["bool_var"].fillna(False).astype("bool[pyarrow]")

        # Adding to the result dataset
        result_dataset.data = final_result

        # Retain only the elements that are specified (True or False)
        if retain_element is not None:
            result_dataset.data = result_dataset.data[
                result_dataset.data["bool_var"] == retain_element
            ]
            result_dataset.data = result_dataset.data.reset_index(drop=True)

        return result_dataset

    @staticmethod
    def _check_all_columns(row: Any) -> bool:
        return all(col_value == True for col_value in row)
