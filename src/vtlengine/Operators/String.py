import math
import operator
from typing import Any, ClassVar, Optional, Union

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
from vtlengine.Model import DataComponent, Dataset, Scalar


class Unary(Operator.Unary):
    type_to_check = String
    str_accessor: Optional[str] = None

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
    def check_param(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def check_param_value(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")


class Substr(Parameterized):
    op = SUBSTR
    return_type = String

    @classmethod
    def validate_params(cls, params: Any) -> None:
        if len(params) != 2:
            raise SemanticError("1-1-18-7", op=cls.op, number=len(params), expected=2)

    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position: int) -> None:
        if not param:
            return
        if position not in (1, 2):
            raise SemanticError("1-1-18-3", op=cls.op, pos=position)
        data_type: Any = param.data_type

        if not check_unary_implicit_promotion(data_type, Integer):
            raise SemanticError("1-1-18-4", op=cls.op, param_type=cls.op, correct_type="Integer")

        if isinstance(param, Scalar):
            cls.check_param_value(param.value, position)

    @classmethod
    def check_param_value(cls, param: Optional[Any], position: int) -> None:
        if param is not None:
            param_is_null = param is None or (isinstance(param, float) and math.isnan(param))
            if not param_is_null and not param >= 1 and position == 1:
                raise SemanticError("1-1-18-4", op=cls.op, param_type="Start", correct_type=">= 1")
            elif not param_is_null and not param >= 0 and position == 2:
                raise SemanticError("1-1-18-4", op=cls.op, param_type="Length", correct_type=">= 0")


class Replace(Parameterized):
    op = REPLACE
    return_type = String

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
        if isinstance(param, Scalar):
            cls.check_param_value(param.value, position)

    @classmethod
    def check_param_value(cls, param: Any, position: int) -> None:
        param_is_null = param is None or (isinstance(param, float) and math.isnan(param))
        if position == 2 and not param_is_null and param < 1:
            raise SemanticError("1-1-18-4", op=cls.op, param_type="Start", correct_type=">= 1")
        elif position == 3 and not param_is_null and param < 1:
            raise SemanticError("1-1-18-4", op=cls.op, param_type="Occurrence", correct_type=">= 1")
