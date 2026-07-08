import operator
import re
from typing import Any, ClassVar, Optional, Union

import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import (
    CONCAT,
    DAMERAU_LEVENSHTEIN,
    HAMMING,
    INSTR,
    JARO_WINKLER,
    LCASE,
    LEN,
    LEVENSHTEIN,
    LTRIM,
    REPLACE,
    RTRIM,
    STRING_DISTANCE,
    SUBSTR,
    TRIM,
    UCASE,
)
from vtlengine.DataTypes import Integer, Number, String, check_unary_implicit_promotion
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Role, Scalar
from vtlengine.ViralPropagation import get_current_registry


class Unary(Operator.Unary):
    type_to_check = String
    str_accessor: Optional[str] = None

    @classmethod
    def op_func(cls, x: Any) -> Any:
        if pd.isnull(x):
            return None
        return cls.py_op(str(x))

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        """Applies the operation to a component"""
        if cls.str_accessor is not None:
            s = (
                series.astype("string[pyarrow]")
                if str(series.dtype) != "string[pyarrow]"
                else series
            )
            return getattr(s.str, cls.str_accessor)()
        return series.map(lambda x: cls.py_op(str(x)), na_action="ignore")

    @classmethod
    def validate_dataset(cls, dataset: Dataset) -> None:
        """
        Validate that the dataset has exactly one measure.
        """
        measures = dataset.get_measures()

        if len(measures) != 1:
            raise SemanticError("1-1-18-1", op=cls.op, name=dataset.name)


class Length(Unary):
    op = LEN
    return_type = Integer
    py_op = len

    @classmethod
    def op_func(cls, x: Any) -> Any:
        if pd.isnull(x):
            return None
        return len(str(x))

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        """Applies the operation to a component"""
        s = series.astype("string[pyarrow]") if str(series.dtype) != "string[pyarrow]" else series
        return s.str.len()


class Lower(Unary):
    op = LCASE
    py_op = str.lower
    return_type = String
    str_accessor = "lower"


class Upper(Unary):
    op = UCASE
    py_op = str.upper
    return_type = String
    str_accessor = "upper"


class Trim(Unary):
    op = TRIM
    py_op = str.strip
    return_type = String
    str_accessor = "strip"


class Ltrim(Unary):
    op = LTRIM
    py_op = str.lstrip
    return_type = String
    str_accessor = "lstrip"


class Rtrim(Unary):
    op = RTRIM
    py_op = str.rstrip
    return_type = String
    str_accessor = "rstrip"


class Binary(Operator.Binary):
    type_to_check = String

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return cls.py_op(str(x), str(y))


class Concatenate(Binary):
    op = CONCAT
    py_op = operator.concat
    return_type = String


class Parameterized(Unary):
    @classmethod
    def validate(cls, *args: Any) -> Any:
        operand: Operator.ALL_MODEL_DATA_TYPES
        param1: Optional[Scalar]
        param2: Optional[Scalar]
        operand, param1, param2 = (args + (None, None))[:3]

        if param1 is not None:
            cls.check_param(param1, 1)
        if param2 is not None:
            cls.check_param(param2, 2)
        return super().validate(operand)

    @classmethod
    def op_func(cls, *args: Any) -> Any:
        x: Optional[Any]
        param1: Optional[Any]
        param2: Optional[Any]
        x, param1, param2 = (args + (None, None))[:3]

        if pd.isnull(x):
            return None
        return cls.py_op(x, param1, param2)

    @classmethod
    def apply_operation_two_series(cls, *args: Any) -> Any:
        left_series, right_series = args

        return left_series.combine(right_series, cls.op_func)

    @classmethod
    def apply_operation_series_scalar(cls, *args: Any) -> Any:
        series, param1, param2 = args

        return series.map(lambda x: cls.op_func(x, param1, param2))

    @classmethod
    def dataset_evaluation(cls, *args: Any) -> Dataset:
        operand: Dataset
        param1: Optional[Union[DataComponent, Scalar]]
        param2: Optional[Union[DataComponent, Scalar]]
        operand, param1, param2 = (args + (None, None))[:3]

        result = cls.validate(operand, param1, param2)
        result.data = operand.data.copy() if operand.data is not None else pd.DataFrame()
        for measure_name in operand.get_measures_names():
            if isinstance(param1, DataComponent) or isinstance(param2, DataComponent):
                result.data[measure_name] = cls.apply_operation_series(
                    result.data[measure_name], param1, param2
                )
            else:
                param_value1 = None if param1 is None else param1.value
                param_value2 = None if param2 is None else param2.value
                result.data[measure_name] = cls.apply_operation_series_scalar(
                    result.data[measure_name], param_value1, param_value2
                )

        cols_to_keep = [c.name for c in operand.get_components() if c.role != Role.ATTRIBUTE]
        result.data = result.data[cols_to_keep]
        # Execute the viral propagation rule on the (row-preserving) result (issue #877).
        get_current_registry().apply_row_preserving(
            result.data, result.get_viral_attributes_names()
        )
        cls.modify_measure_column(result)
        return result

    @classmethod
    def component_evaluation(cls, *args: Any) -> DataComponent:
        operand: DataComponent
        param1: Optional[Union[DataComponent, Scalar]]
        param2: Optional[Union[DataComponent, Scalar]]
        operand, param1, param2 = (args + (None, None))[:3]

        result = cls.validate(operand, param1, param2)
        result.data = operand.data.copy() if operand.data is not None else pd.Series()
        if isinstance(param1, DataComponent) or isinstance(param2, DataComponent):
            result.data = cls.apply_operation_series(result.data, param1, param2)
        else:
            param_value1 = None if param1 is None else param1.value
            param_value2 = None if param2 is None else param2.value
            result.data = cls.apply_operation_series_scalar(
                operand.data, param_value1, param_value2
            )
        return result

    @classmethod
    def scalar_evaluation(cls, *args: Any) -> Scalar:
        operand: Scalar
        param1: Optional[Scalar]
        param2: Optional[Scalar]
        operand, param1, param2 = (args + (None, None))[:3]

        result = cls.validate(operand, param1, param2)
        param_value1 = None if param1 is None else param1.value
        param_value2 = None if param2 is None else param2.value
        result.value = cls.op_func(operand.value, param_value1, param_value2)
        return result

    @classmethod
    def evaluate(cls, *args: Any) -> Union[Dataset, DataComponent, Scalar]:
        param1: Optional[Union[DataComponent, Scalar]]
        param2: Optional[Union[DataComponent, Scalar]]
        operand, param1, param2 = (args + (None, None))[:3]

        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, param1, param2)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, param1, param2)
        return cls.scalar_evaluation(operand, param1, param2)

    @classmethod
    def check_param(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def check_param_value(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def generate_series_from_param(cls, *args: Any) -> Any:
        param: Optional[Union[DataComponent, Scalar]] = None
        length: int
        if len(args) == 2:
            param, length = args
        else:
            length = args[0]

        if param is None:
            return pd.Series(index=range(length), dtype="string[pyarrow]")
        if isinstance(param, Scalar):
            return pd.Series(data=param.value, index=range(length), dtype="string[pyarrow]")
        return param.data

    @classmethod
    def apply_operation_series(cls, *args: Any) -> Any:
        param1: Optional[Union[DataComponent, Scalar]]
        param2: Optional[Union[DataComponent, Scalar]]
        data, param1, param2 = (args + (None, None))[:3]

        param1_data = cls.generate_series_from_param(param1, len(data))
        param2_data = cls.generate_series_from_param(param2, len(data))
        df = pd.DataFrame([data, param1_data, param2_data]).T
        n1, n2, n3 = df.columns
        return df.apply(lambda x: cls.op_func(x[n1], x[n2], x[n3]), axis=1)


class Substr(Parameterized):
    op = SUBSTR
    return_type = String

    @classmethod
    def validate_params(cls, params: Any) -> None:
        if len(params) != 2:
            raise SemanticError("1-1-18-7", op=cls.op, number=len(params), expected=2)

    @classmethod
    def py_op(cls, x: str, param1: Any, param2: Any) -> Any:
        x = str(x)
        param1 = None if pd.isnull(param1) else int(param1)
        param2 = None if pd.isnull(param2) else int(param2)
        if param1 is None and param2 is None:
            return x
        if param1 is None:
            param1 = 0
        elif param1 != 0:
            param1 -= 1
        elif param1 > (len(x)):
            return ""
        param2 = len(x) if param2 is None or param1 + param2 > len(x) else param1 + param2
        return x[param1:param2]

    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position: int) -> None:
        if not param:
            return
        if position not in (1, 2):
            raise SemanticError("1-1-18-3", op=cls.op, pos=position)
        data_type: Any = param.data_type

        if not check_unary_implicit_promotion(data_type, Integer):
            raise SemanticError("1-1-18-4", op=cls.op, param_type=cls.op, correct_type="Integer")

        if isinstance(param, DataComponent):
            if param.data is not None:
                param.data.map(lambda x: cls.check_param_value(x, position))
        else:
            cls.check_param_value(param.value, position)

    @classmethod
    def check_param_value(cls, param: Optional[Any], position: int) -> None:
        if param is not None:
            if not pd.isnull(param) and not param >= 1 and position == 1:
                raise SemanticError("1-1-18-4", op=cls.op, param_type="Start", correct_type=">= 1")
            elif not pd.isnull(param) and not param >= 0 and position == 2:
                raise SemanticError("1-1-18-4", op=cls.op, param_type="Length", correct_type=">= 0")


class Replace(Parameterized):
    op = REPLACE
    return_type = String

    @classmethod
    def py_op(cls, x: str, param1: Optional[Any], param2: Optional[Any]) -> Any:
        if pd.isnull(param1):
            return None
        if pd.isnull(param2):
            return None
        x = str(x)
        return x.replace(str(param1), str(param2))

    @classmethod
    def evaluate(cls, *args: Any) -> Union[Dataset, DataComponent, Scalar]:
        operand, param1, param2 = (args + (None, None))[:3]
        if param2 is None:
            param2 = Scalar(name="replace_default", data_type=String, value="")
        return super().evaluate(operand, param1, param2)

    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position: int) -> None:
        if not param:
            return
        if position not in (1, 2):
            raise SemanticError("1-1-18-3", op=cls.op, pos=position)
        data_type: Any = param.data_type

        if not check_unary_implicit_promotion(data_type, String):
            raise SemanticError("1-1-18-4", op=cls.op, param_type=cls.op, correct_type="String")

    @classmethod
    def validate_params(cls, params: Any) -> None:
        if len(params) != 2:
            raise SemanticError("1-1-18-7", op=cls.op, number=len(params), expected=2)


class StringDistance(Binary):
    """Base class for the four `string_distance(method, s1, s2)` variants."""

    op = STRING_DISTANCE
    type_to_check = String
    return_type = Number
    method_name: ClassVar[str]

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return cls.py_op(str(x), str(y))


class Levenshtein(StringDistance):
    """Levenshtein distance — minimum single-character edits to turn s1 into s2."""

    method_name = LEVENSHTEIN

    @staticmethod
    def py_op(s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        if len(s1) == 0:
            return len(s2)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, start=1):
            curr = [i] + [0] * len(s2)
            for j, c2 in enumerate(s2, start=1):
                cost = 0 if c1 == c2 else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[-1]


class DamerauLevenshtein(StringDistance):
    """Damerau-Levenshtein distance — Levenshtein with adjacent transpositions as 1 edit."""

    method_name = DAMERAU_LEVENSHTEIN

    @staticmethod
    def py_op(s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
        return dp[len1][len2]


class Hamming(StringDistance):
    """Hamming distance — count of differing positions. Requires equal length."""

    method_name = HAMMING

    @staticmethod
    def py_op(s1: str, s2: str) -> int:
        if len(s1) != len(s2):
            raise SemanticError("1-1-18-11", op=STRING_DISTANCE, len1=len(s1), len2=len(s2))
        return sum(1 for a, b in zip(s1, s2) if a != b)


class JaroWinkler(StringDistance):
    """Jaro-Winkler similarity in [0, 1] (1.0 == identical)."""

    method_name = JARO_WINKLER

    @staticmethod
    def py_op(s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        match_distance = max(0, max(len1, len2) // 2 - 1)
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        transpositions = 0
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        transpositions //= 2
        jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3
        prefix = 0
        for i in range(min(4, len1, len2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        # 0.1 is the standard Winkler prefix scaling factor (per Wikipedia / GfG).
        return jaro + prefix * 0.1 * (1 - jaro)


DISTANCE_DISPATCH: dict[str, type[StringDistance]] = {
    cls.method_name: cls for cls in StringDistance.__subclasses__()
}


class Instr(Parameterized):
    op = INSTR
    return_type = Integer

    @classmethod
    def validate(
        cls,
        operand: Operator.ALL_MODEL_DATA_TYPES,
        param1: Optional[Operator.ALL_MODEL_DATA_TYPES] = None,
        param2: Optional[Operator.ALL_MODEL_DATA_TYPES] = None,
        param3: Optional[Operator.ALL_MODEL_DATA_TYPES] = None,
    ) -> Any:
        if (
            isinstance(param1, Dataset)
            or isinstance(param2, Dataset)
            or isinstance(param3, Dataset)
        ):
            raise SemanticError("1-1-18-10", op=cls.op)
        if param1 is not None:
            cls.check_param(param1, 1)
        if param2 is not None:
            cls.check_param(param2, 2)
        if param3 is not None:
            cls.check_param(param3, 3)

        return super().validate(operand)

    @classmethod
    def validate_params(cls, params: Any) -> None:
        if len(params) != 2:
            raise SemanticError("1-1-18-7", op=cls.op, number=len(params), expected=2)

    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position: int) -> None:
        if not param:
            return
        if position not in (1, 2, 3):
            raise SemanticError("1-1-18-9", op=cls.op)
        data_type: Any = param.data_type

        if position == 1:
            if not check_unary_implicit_promotion(data_type, String):
                raise SemanticError(
                    "1-1-18-4", op=cls.op, param_type="Pattern", correct_type="String"
                )
        elif position == 2:
            if not check_unary_implicit_promotion(data_type, Integer):
                raise SemanticError(
                    "1-1-18-4", op=cls.op, param_type="Start", correct_type="Integer"
                )
        else:
            if not check_unary_implicit_promotion(data_type, Integer):
                raise SemanticError(
                    "1-1-18-4",
                    op=cls.op,
                    param_type="Occurrence",
                    correct_type="Integer",
                )
        if isinstance(param, DataComponent):
            if param.data is not None:
                param.data.map(lambda x: cls.check_param_value(x, position))
        else:
            cls.check_param_value(param.value, position)

    @classmethod
    def check_param_value(cls, param: Any, position: int) -> None:
        if position == 2 and not pd.isnull(param) and param < 1:
            raise SemanticError("1-1-18-4", op=cls.op, param_type="Start", correct_type=">= 1")
        elif position == 3 and not pd.isnull(param) and param < 1:
            raise SemanticError("1-1-18-4", op=cls.op, param_type="Occurrence", correct_type=">= 1")

    @classmethod
    def apply_operation_series_scalar(
        cls, series: Any, param1: Any, param2: Any, param3: Any
    ) -> Any:
        return series.map(lambda x: cls.op_func(x, param1, param2, param3))

    @classmethod
    def apply_operation_series(
        cls,
        data: Any,
        param1: Optional[Union[DataComponent, Scalar]],
        param2: Optional[Union[DataComponent, Scalar]],
        param3: Optional[Union[DataComponent, Scalar]],
    ) -> Any:
        param1_data = cls.generate_series_from_param(param1, len(data))
        param2_data = cls.generate_series_from_param(param2, len(data))
        param3_data = cls.generate_series_from_param(param3, len(data))

        df = pd.DataFrame([data, param1_data, param2_data, param3_data]).T
        n1, n2, n3, n4 = df.columns
        return df.apply(lambda x: cls.op_func(x[n1], x[n2], x[n3], x[n4]), axis=1)

    @classmethod
    def dataset_evaluation(  # type: ignore[override]
        cls,
        operand: Dataset,
        param1: Optional[Union[DataComponent, Scalar]],
        param2: Optional[Union[DataComponent, Scalar]],
        param3: Optional[Union[DataComponent, Scalar]],
    ) -> Dataset:
        result = cls.validate(operand, param1, param2, param3)
        result.data = operand.data.copy() if operand.data is not None else pd.DataFrame()
        for measure_name in operand.get_measures_names():
            if (
                isinstance(param1, DataComponent)
                or isinstance(param2, DataComponent)
                or isinstance(param3, DataComponent)
            ):
                if operand.data is not None:
                    result.data[measure_name] = cls.apply_operation_series(
                        operand.data[measure_name], param1, param2, param3
                    )
            else:
                param_value1 = None if param1 is None else param1.value
                param_value2 = None if param2 is None else param2.value
                param_value3 = None if param3 is None else param3.value
                result.data[measure_name] = cls.apply_operation_series_scalar(
                    result.data[measure_name], param_value1, param_value2, param_value3
                )
        cols_to_keep = [c.name for c in operand.get_components() if c.role != Role.ATTRIBUTE]
        result.data = result.data[cols_to_keep]
        # Execute the viral propagation rule on the (row-preserving) result (issue #877).
        get_current_registry().apply_row_preserving(
            result.data, result.get_viral_attributes_names()
        )
        cls.modify_measure_column(result)
        return result

    @classmethod
    def component_evaluation(  # type: ignore[override]
        cls,
        operand: DataComponent,
        param1: Optional[Union[DataComponent, Scalar]],
        param2: Optional[Union[DataComponent, Scalar]],
        param3: Optional[Union[DataComponent, Scalar]],
    ) -> DataComponent:
        result = cls.validate(operand, param1, param2, param3)
        result.data = operand.data.copy() if operand.data is not None else pd.Series()
        if (
            isinstance(param1, DataComponent)
            or isinstance(param2, DataComponent)
            or isinstance(param3, DataComponent)
        ):
            result.data = cls.apply_operation_series(operand.data, param1, param2, param3)
        else:
            param_value1 = None if param1 is None else param1.value
            param_value2 = None if param2 is None else param2.value
            param_value3 = None if param3 is None else param3.value
            result.data = cls.apply_operation_series_scalar(
                operand.data, param_value1, param_value2, param_value3
            )
        return result

    @classmethod
    def scalar_evaluation(  # type: ignore[override]
        cls,
        operand: Scalar,
        param1: Optional[Scalar],
        param2: Optional[Scalar],
        param3: Optional[Scalar],
    ) -> Scalar:
        result = cls.validate(operand, param1, param2, param3)
        param_value1 = None if param1 is None else param1.value
        param_value2 = None if param2 is None else param2.value
        param_value3 = None if param3 is None else param3.value
        result.value = cls.op_func(operand.value, param_value1, param_value2, param_value3)
        return result

    @classmethod
    def evaluate(
        cls,
        operand: Operator.ALL_MODEL_DATA_TYPES,
        param1: Optional[Any] = None,
        param2: Optional[Any] = None,
        param3: Optional[Any] = None,
    ) -> Any:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, param1, param2, param3)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, param1, param2, param3)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand, param1, param2, param3)

    @classmethod
    def op_func(  # type: ignore[override]
        cls,
        x: Any,
        param1: Optional[Any],
        param2: Optional[Any],
        param3: Optional[Any],
    ) -> Any:
        if pd.isnull(x):
            return None
        return cls.py_op(x, param1, param2, param3)

    @classmethod
    def py_op(
        cls,
        str_value: str,
        str_to_find: Optional[str],
        start: Optional[int],
        occurrence: Optional[int],
    ) -> Any:
        str_value = str(str_value)
        if not pd.isnull(start):
            if isinstance(start, (int, float)):
                start = int(start - 1)
            else:
                # OPERATORS_STRINGOPERATORS.92
                raise SemanticError(
                    "1-1-18-4", op=cls.op, param_type="Start", correct_type="Integer"
                )
        else:
            start = 0

        if not pd.isnull(occurrence):
            if isinstance(occurrence, (int, float)):
                occurrence = int(occurrence - 1)
            else:
                # OPERATORS_STRINGOPERATORS.93
                raise SemanticError(
                    "1-1-18-4",
                    op=cls.op,
                    param_type="Occurrence",
                    correct_type="Integer",
                )
        else:
            occurrence = 0
        if pd.isnull(str_to_find):
            return None
        else:
            str_to_find = str(str_to_find)

        occurrences_list = [m.start() for m in re.finditer(str_to_find, str_value[start:])]

        length = len(occurrences_list)

        position = 0 if occurrence > length - 1 else int(start + occurrences_list[occurrence] + 1)

        return position
