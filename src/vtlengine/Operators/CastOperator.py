import re
from copy import copy
from datetime import datetime
from typing import Any, Optional, Type, Union

import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import CAST
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    EXPLICIT_WITH_MASK_TYPE_PROMOTION_MAPPING,
    EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING,
    IMPLICIT_TYPE_PROMOTION_MAPPING,
    SCALAR_TYPES_CLASS_REVERSE,
    Date,
    Duration,
    Integer,
    Number,
    ScalarType,
    String,
    TimeInterval,
    TimePeriod,
)
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler, period_to_date
from vtlengine.Exceptions import RunTimeError, SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar
from vtlengine.Utils.__Virtual_Assets import VirtualCounter

duration_mapping = {"A": 6, "S": 5, "Q": 4, "M": 3, "W": 2, "D": 1}

# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


_NUM_MASK_RE = re.compile(r"D+|d+|[Ee][+\-]?[DdEe]*|[+\-]|.")


def _parse_vtl_number_mask(mask: str) -> str:
    """Convert a VTL number mask (e.g. 'DD.DDD') to a regex pattern string."""
    parts: list[str] = []
    for tok in _NUM_MASK_RE.findall(mask):
        c = tok[0]
        if c == "D":
            parts.append(rf"\d{{{len(tok)}}}")
        elif c == "d":
            parts.append(rf"\d{{0,{len(tok)}}}")
        elif c in ("E", "e"):
            parts.append(r"[Ee][+-]?\d+")
        elif c == ".":
            parts.append(r"[^\d]")
        elif c == ",":
            parts.append(",")
        elif c in ("+", "-"):
            continue
        else:
            parts.append(re.escape(tok))
    return rf"^[+-]?{''.join(parts)}$"


_VTL_DATE_TOKENS = [
    ("YYYY", "%Y"),
    ("MONTH", "%B"),
    ("Month", "%B"),
    ("month", "%B"),
    ("MON", "%b"),
    ("DAY", "%A"),
    ("Day", "%A"),
    ("day", "%a"),
    ("YY", "%y"),
    ("MM", "%m"),
    ("DD", "%d"),
    ("hh", "%H"),
    ("mm", "%M"),
    ("ss", "%S"),
]
_VTL_TOKEN_RE = re.compile("|".join(re.escape(t[0]) for t in _VTL_DATE_TOKENS))


def _vtl_date_mask_to_python(mask: str) -> str:
    """Convert a VTL date mask to a Python strptime/strftime format string."""
    lookup = dict(_VTL_DATE_TOKENS)
    return _VTL_TOKEN_RE.sub(lambda m: lookup[m.group(0)], mask)


_TP_MASK_RE = re.compile(r"YYYY|YY|\\.|Q+|S+|A+|W+|M+|D+|.")


def _parse_vtl_tp_mask(mask: str) -> list[dict[str, Any]]:
    """
    Tokenize a VTL TimePeriod mask into a list of token dicts.

    Token types:
      year       - {'type':'year', 'n': 4|2}
      literal    - {'type':'literal', 'ch': str}
      period_num - {'type':'period_num', 'indicator': str, 'n': int}
      cal_month  - {'type':'cal_month', 'n': int}
      cal_day    - {'type':'cal_day',   'n': int}
    """
    tokens: list[dict[str, Any]] = []
    for m in _TP_MASK_RE.finditer(mask):
        tok = m.group()
        if tok in ("YYYY", "YY"):
            tokens.append({"type": "year", "n": len(tok)})
        elif tok.startswith("\\"):
            tokens.append({"type": "literal", "ch": tok[1]})
        elif tok[0] in "QSAW":
            tokens.append({"type": "period_num", "indicator": tok[0], "n": len(tok)})
        elif tok[0] == "M":
            if "D" in mask[m.end() :]:
                tokens.append({"type": "cal_month", "n": len(tok)})
            else:
                tokens.append({"type": "period_num", "indicator": "M", "n": len(tok)})
        elif tok[0] == "D":
            has_m = any(
                t["type"] == "cal_month" or (t["type"] == "period_num" and t["indicator"] == "M")
                for t in tokens
            )
            if has_m:
                tokens.append({"type": "cal_day", "n": len(tok)})
            else:
                tokens.append({"type": "period_num", "indicator": "D", "n": len(tok)})
        else:
            tokens.append({"type": "literal", "ch": tok})
    return tokens


def _infer_tp_period_type(tokens: list[dict[str, Any]]) -> str:
    """Infer the VTL period type (A/S/Q/M/W/D or D_CAL) from a tokenized mask."""
    pn_inds = [t["indicator"] for t in tokens if t["type"] == "period_num"]
    has_cal_month = any(t["type"] == "cal_month" for t in tokens)
    has_cal_day = any(t["type"] == "cal_day" for t in tokens)

    if has_cal_month and has_cal_day:
        return "D_CAL"
    if has_cal_month:
        return "M"
    if "M" in pn_inds and "D" in pn_inds:
        return "D_CAL"  # e.g. YYYY\MMM\DDD

    for ind in ("Q", "S", "W", "M", "D", "A"):
        if ind in pn_inds:
            return ind
    return "A"


ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]


class Cast(Operator.Unary):
    op = CAST

    # CASTS VALUES
    # Converts the value from one type to another according to the mask
    @classmethod
    def cast_string_to_integer(cls, value: Any, mask: str) -> Any:
        """Cast a string to an integer, according to the mask."""
        pattern = _parse_vtl_number_mask(mask)
        stripped = str(value).strip()
        if not re.match(pattern, stripped):
            raise RunTimeError(
                "2-1-5-3",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[String],
                type_2=SCALAR_TYPES_CLASS_REVERSE[Integer],
                mask=mask,
            )
        return int(float(stripped.replace(",", "")))

    @classmethod
    def cast_string_to_number(cls, value: Any, mask: str) -> Any:
        """Cast a string to a number, according to the mask."""
        pattern = _parse_vtl_number_mask(mask)
        stripped = str(value).strip()
        if not re.match(pattern, stripped):
            raise RunTimeError(
                "2-1-5-3",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[String],
                type_2=SCALAR_TYPES_CLASS_REVERSE[Number],
                mask=mask,
            )
        # Normalize: replace any non-digit separator (between digits) with '.' for float()
        normalized = re.sub(r"(?<=\d)[^\deE+\-](?=\d)", ".", stripped)
        return float(normalized)

    @classmethod
    def cast_string_to_date(cls, value: Any, mask: str) -> Any:
        """Cast a string to a date, according to the mask."""
        py_fmt = _vtl_date_mask_to_python(mask)
        stripped = str(value).strip()
        try:
            return datetime.strptime(stripped, py_fmt).date().isoformat()
        except ValueError:
            raise RunTimeError(
                "2-1-5-3",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[String],
                type_2=SCALAR_TYPES_CLASS_REVERSE[Date],
                mask=mask,
            )

    @classmethod
    def cast_string_to_duration(cls, value: Any, mask: str) -> Any:
        """Cast a string to a duration, according to the mask."""
        from vtlengine.DataTypes import _ISO_TO_SHORTCODE
        from vtlengine.DataTypes.TimeHandling import PERIOD_IND_MAPPING

        stripped = str(value).strip().upper()
        if stripped.startswith("P"):
            shortcode = _ISO_TO_SHORTCODE.get(stripped)
            if shortcode is not None:
                return shortcode
        elif stripped in PERIOD_IND_MAPPING:
            return stripped
        raise RunTimeError(
            "2-1-5-3",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[String],
            type_2=SCALAR_TYPES_CLASS_REVERSE[Duration],
            mask=mask,
        )

    @classmethod
    def cast_string_to_time_period(cls, value: Any, mask: str) -> Any:
        """Cast a string to a time period, according to the mask."""
        from datetime import date as date_cls

        _tp_error_kwargs = {
            "value": value,
            "type_1": SCALAR_TYPES_CLASS_REVERSE[String],
            "type_2": SCALAR_TYPES_CLASS_REVERSE[TimePeriod],
            "mask": mask,
        }

        tokens = _parse_vtl_tp_mask(mask)
        period_type = _infer_tp_period_type(tokens)
        s = str(value).strip()
        pos = 0
        year: Optional[int] = None
        period_num: Optional[int] = None
        cal_month: Optional[int] = None
        cal_day: Optional[int] = None

        for t in tokens:
            if pos > len(s):
                raise RunTimeError("2-1-5-3", **_tp_error_kwargs)
            ttype = t["type"]
            if ttype == "year":
                n = t["n"]
                chunk = s[pos : pos + n]
                if not chunk.isdigit():
                    raise RunTimeError("2-1-5-3", **_tp_error_kwargs)
                year = int(chunk) + (2000 if n == 2 else 0)
                pos += n
            elif ttype == "literal":
                if pos >= len(s) or s[pos] != t["ch"]:
                    raise RunTimeError("2-1-5-3", **_tp_error_kwargs)
                pos += 1
            elif ttype == "period_num":
                n = t["n"]
                chunk = s[pos : pos + n]
                if not chunk.isdigit():
                    raise RunTimeError("2-1-5-3", **_tp_error_kwargs)
                ind = t["indicator"]
                if period_type == "D_CAL" and ind == "M":
                    cal_month = int(chunk)
                elif period_type == "D_CAL" and ind == "D":
                    cal_day = int(chunk)
                else:
                    period_num = int(chunk)
                pos += n
            elif ttype == "cal_month":
                n = t["n"]
                chunk = s[pos : pos + n]
                if not chunk.isdigit():
                    raise RunTimeError("2-1-5-3", **_tp_error_kwargs)
                cal_month = int(chunk)
                pos += n
            elif ttype == "cal_day":
                n = t["n"]
                chunk = s[pos : pos + n]
                if not chunk.isdigit():
                    raise RunTimeError("2-1-5-3", **_tp_error_kwargs)
                cal_day = int(chunk)
                pos += n

        if pos != len(s) or year is None:
            raise RunTimeError("2-1-5-3", **_tp_error_kwargs)

        if period_type in ("D_CAL",) or (cal_month is not None and cal_day is not None):
            d = date_cls(year, cal_month, cal_day)  # type: ignore[arg-type]
            doy = d.timetuple().tm_yday
            return str(TimePeriodHandler(f"{year}D{doy}"))
        if period_type == "A" or period_num is None:
            return str(TimePeriodHandler(f"{year}A"))
        return str(TimePeriodHandler(f"{year}{period_type}{period_num}"))

    @classmethod
    def cast_string_to_time(cls, value: Any, mask: str) -> Any:
        """Cast a string to a time (TimeInterval), according to the mask."""
        if "/" not in mask:
            return str(value).strip()
        mask_parts = mask.split("/", 1)
        val_parts = str(value).strip().split("/", 1)
        if len(val_parts) != 2:
            raise RunTimeError(
                "2-1-5-3",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[String],
                type_2=SCALAR_TYPES_CLASS_REVERSE[TimeInterval],
                mask=mask,
            )
        date1 = cls.cast_string_to_date(val_parts[0].strip(), mask_parts[0])
        date2 = cls.cast_string_to_date(val_parts[1].strip(), mask_parts[1])
        return f"{date1}/{date2}"

    @classmethod
    def cast_integer_to_string(cls, value: Any, mask: str) -> Any:
        """Cast an integer to a string, according to the mask."""
        return cls.cast_number_to_string(float(value), mask)

    @classmethod
    def cast_number_to_string(cls, value: Any, mask: str) -> Any:
        """Cast a number to a string, according to the mask."""
        fval = float(value)
        # Count D-groups on each side of the decimal separator
        sep = "." if "." in mask else ("," if "," in mask else None)
        if sep:
            idx = mask.index(sep)
            n_dec = mask[idx + 1 :].count("D") + mask[idx + 1 :].count("d")
            n_int = mask[:idx].count("D") + mask[:idx].count("d")
        else:
            n_dec = 0
            n_int = mask.count("D") + mask.count("d")

        negative = fval < 0
        abs_val = abs(fval)
        if n_dec > 0:
            formatted = f"{abs_val:.{n_dec}f}"
            int_str, dec_str = formatted.split(".")
            int_str = int_str.zfill(n_int)
            result = int_str + sep + dec_str  # type: ignore[operator]
        else:
            result = str(int(abs_val)).zfill(n_int)
        return ("-" if negative else "") + result

    @classmethod
    def cast_date_to_string(cls, value: Any, mask: str) -> Any:
        """Cast a date to a string, according to the mask."""
        py_fmt = _vtl_date_mask_to_python(mask)
        try:
            return datetime.fromisoformat(str(value)).strftime(py_fmt)
        except ValueError:
            raise RunTimeError(
                "2-1-5-3",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[Date],
                type_2=SCALAR_TYPES_CLASS_REVERSE[String],
                mask=mask,
            )

    @classmethod
    def cast_time_to_string(cls, value: Any, mask: str) -> Any:
        """Cast a time (TimeInterval) to a string, according to the mask."""
        if "/" not in mask:
            return str(value)
        mask_parts = mask.split("/", 1)
        val_str = str(value)
        if "/" not in val_str:
            raise RunTimeError(
                "2-1-5-3",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[TimeInterval],
                type_2=SCALAR_TYPES_CLASS_REVERSE[String],
                mask=mask,
            )
        val_parts = val_str.split("/", 1)
        date1 = cls.cast_date_to_string(val_parts[0], mask_parts[0])
        date2 = cls.cast_date_to_string(val_parts[1], mask_parts[1])
        return f"{date1}/{date2}"

    @classmethod
    def cast_time_period_to_string(cls, value: Any, mask: str) -> Any:
        """Cast a time_period to a string, according to the mask."""
        handler = TimePeriodHandler(str(value))
        tokens = _parse_vtl_tp_mask(mask)
        period_type = _infer_tp_period_type(tokens)
        parts: list[str] = []

        for t in tokens:
            ttype = t["type"]
            if ttype == "year":
                parts.append(f"{handler.year:04d}" if t["n"] == 4 else f"{handler.year % 100:02d}")
            elif ttype == "literal":
                parts.append(t["ch"])
            elif ttype == "period_num":
                n = t["n"]
                ind = t["indicator"]
                if period_type == "D_CAL":
                    start = period_to_date(
                        handler.year, handler.period_indicator, handler.period_number, start=True
                    )
                    val = start.month if ind == "M" else start.day
                else:
                    val = handler.period_number
                parts.append(f"{val:0{n}d}")
            elif ttype == "cal_month":
                start = period_to_date(
                    handler.year, handler.period_indicator, handler.period_number, start=True
                )
                parts.append(f"{start.month:0{t['n']}d}")
            elif ttype == "cal_day":
                start = period_to_date(
                    handler.year, handler.period_indicator, handler.period_number, start=True
                )
                parts.append(f"{start.day:0{t['n']}d}")

        return "".join(parts)

    @classmethod
    def cast_duration_to_string(cls, value: Any, mask: str) -> Any:
        """Cast a duration to a string, according to the mask (ISO-8601)."""
        from vtlengine.DataTypes import _SHORTCODE_TO_ISO

        return _SHORTCODE_TO_ISO.get(str(value), str(value))

    invalid_mask_message = "At op {op}: Invalid mask to cast from type {type_1} to {type_2}."

    @classmethod
    def check_mask_value(
        cls, from_type: Type[ScalarType], to_type: Type[ScalarType], mask_value: str
    ) -> None:
        """
        This method checks if the mask value is valid for the cast operation.
        """
        valid_types = (Integer, Number, TimeInterval, Date, TimePeriod, Duration)

        # from = String
        if (from_type == String and to_type in valid_types) or (
            to_type == String and from_type in valid_types
        ):
            return

        raise SemanticError(
            "1-1-5-5",
            op=cls.op,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
            mask_value=mask_value,
        )

    @classmethod
    def check_cast(
        cls,
        from_type: Type[ScalarType],
        to_type: Type[ScalarType],
        mask_value: Optional[str],
    ) -> None:
        if mask_value is not None:
            cls.check_with_mask(from_type, to_type, mask_value)
        else:
            cls.check_without_mask(from_type, to_type)

    @classmethod
    def check_with_mask(
        cls, from_type: Type[ScalarType], to_type: Type[ScalarType], mask_value: str
    ) -> None:
        explicit_promotion = EXPLICIT_WITH_MASK_TYPE_PROMOTION_MAPPING[from_type]
        if to_type.is_included(explicit_promotion):
            return cls.check_mask_value(from_type, to_type, mask_value)

        raise SemanticError(
            "1-1-5-5",
            op=cls.op,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
            mask_value=mask_value,
        )

    @classmethod
    def check_without_mask(cls, from_type: Type[ScalarType], to_type: Type[ScalarType]) -> None:
        explicit_promotion = EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING[from_type]
        implicit_promotion = IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]
        if not (to_type.is_included(explicit_promotion) or to_type.is_included(implicit_promotion)):
            explicit_with_mask = EXPLICIT_WITH_MASK_TYPE_PROMOTION_MAPPING[from_type]
            if to_type.is_included(explicit_with_mask):
                raise SemanticError(
                    "1-1-5-3",
                    op=cls.op,
                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                    type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
                )
            raise SemanticError(
                "1-1-5-4",
                op=cls.op,
                type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
            )

    @classmethod
    def cast_component(
        cls, data: Any, from_type: Type[ScalarType], to_type: Type[ScalarType]
    ) -> Any:
        """
        Cast the component to the type to_type without mask
        """

        if to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
            result = data.map(lambda x: to_type.implicit_cast(x, from_type), na_action="ignore")
        else:
            result = data.map(lambda x: to_type.explicit_cast(x, from_type), na_action="ignore")
        return result

    @classmethod
    def cast_mask_component(cls, data: Any, from_type: Any, to_type: Any, mask: str) -> Any:
        result = data.map(lambda x: cls.cast_value(x, from_type, to_type, mask), na_action="ignore")
        return result

    @classmethod
    def cast_value(
        cls,
        value: Any,
        provided_type: Type[ScalarType],
        to_type: Type[ScalarType],
        mask_value: str,
    ) -> Any:
        # from = String
        if provided_type == String and to_type == Integer:
            return cls.cast_string_to_integer(value, mask_value)
        if provided_type == String and to_type == Number:
            return cls.cast_string_to_number(value, mask_value)
        if provided_type == String and to_type == Date:
            return cls.cast_string_to_date(value, mask_value)
        if provided_type == String and to_type == Duration:
            return cls.cast_string_to_duration(value, mask_value)
        if provided_type == String and to_type == TimePeriod:
            return cls.cast_string_to_time_period(value, mask_value)
        if provided_type == String and to_type == TimeInterval:
            return cls.cast_string_to_time(value, mask_value)
        # to = String
        if provided_type == Integer and to_type == String:
            return cls.cast_integer_to_string(value, mask_value)
        if provided_type == Number and to_type == String:
            return cls.cast_number_to_string(value, mask_value)
        if provided_type == Date and to_type == String:
            return cls.cast_date_to_string(value, mask_value)
        if provided_type == Duration and to_type == String:
            return cls.cast_duration_to_string(value, mask_value)
        if provided_type == TimeInterval and to_type == String:
            return cls.cast_time_to_string(value, mask_value)
        if provided_type == TimePeriod and to_type == String:
            return cls.cast_time_period_to_string(value, mask_value)

        raise SemanticError(
            "2-1-5-1",
            op=cls.op,
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[provided_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
        )

    @classmethod
    def validate(  # type: ignore[override]
        cls,
        operand: ALL_MODEL_DATA_TYPES,
        scalarType: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Any:
        if mask is not None and not isinstance(mask, str):
            raise Exception(f"{cls.op} mask must be a string")

        if isinstance(operand, Dataset):
            return cls.dataset_validation(operand, scalarType, mask)
        elif isinstance(operand, DataComponent):
            return cls.component_validation(operand, scalarType, mask)
        elif isinstance(operand, Scalar):
            return cls.scalar_validation(operand, scalarType, mask)

    @classmethod
    def dataset_validation(  # type: ignore[override]
        cls,
        operand: Dataset,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Dataset:
        """
        This method validates the operation when the operand is a Dataset.
        """

        # monomeasure
        if len(operand.get_measures()) != 1:
            raise Exception(f"{cls.op} can only be applied to a Dataset with one measure")
        measure = operand.get_measures()[0]
        from_type = measure.data_type

        cls.check_cast(from_type, to_type, mask)
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in operand.components.items()
            if comp.role != Role.MEASURE
        }

        if not to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
            measure_name = COMP_NAME_MAPPING[to_type]
        else:
            measure_name = measure.name
        result_components[measure_name] = Component(
            name=measure_name,
            data_type=to_type,
            role=Role.MEASURE,
            nullable=measure.nullable,
        )
        dataset_name = VirtualCounter._new_ds_name()
        return Dataset(name=dataset_name, components=result_components, data=None)

    @classmethod
    def component_validation(  # type: ignore[override]
        cls,
        operand: DataComponent,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> DataComponent:
        """
        This method validates the operation when the operand is a DataComponent.
        """

        from_type = operand.data_type
        cls.check_cast(from_type, to_type, mask)
        comp_name = VirtualCounter._new_dc_name()
        return DataComponent(
            name=comp_name,
            data=None,
            data_type=to_type,
            role=operand.role,
            nullable=operand.nullable,
        )

    @classmethod
    def scalar_validation(  # type: ignore[override]
        cls,
        operand: Scalar,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Scalar:
        """
        This method validates the operation when the operand is a DataComponent.
        """

        from_type = operand.data_type
        cls.check_cast(from_type, to_type, mask)
        return Scalar(name=operand.name, data_type=to_type, value=None)

    @classmethod
    def evaluate(  # type: ignore[override]
        cls,
        operand: ALL_MODEL_DATA_TYPES,
        scalarType: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Any:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, scalarType, mask)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand, scalarType, mask)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, scalarType, mask)

    @classmethod
    def dataset_evaluation(  # type: ignore[override]
        cls,
        operand: Dataset,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Dataset:
        from_type = operand.get_measures()[0].data_type
        original_measure = operand.get_measures()[0]
        result_dataset = cls.dataset_validation(operand, to_type, mask)
        new_measure = result_dataset.get_measures()[0]
        result_dataset.data = operand.data.copy() if operand.data is not None else pd.DataFrame()

        if original_measure.name != new_measure.name:
            result_dataset.data.rename(
                columns={original_measure.name: new_measure.name}, inplace=True
            )
        measure_data = result_dataset.data[new_measure.name]
        if mask:
            result_dataset.data[new_measure.name] = cls.cast_mask_component(
                measure_data, from_type, to_type, mask
            )
        else:
            result_dataset.data[new_measure.name] = cls.cast_component(
                measure_data, from_type, to_type
            )
        return result_dataset

    @classmethod
    def scalar_evaluation(  # type: ignore[override]
        cls,
        operand: Scalar,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Scalar:
        from_type = operand.data_type
        result_scalar = cls.scalar_validation(operand, to_type, mask)
        if pd.isna(operand.value):
            return Scalar(name=result_scalar.name, data_type=to_type, value=None)
        if mask:
            casted_data = cls.cast_value(operand.value, operand.data_type, to_type, mask)
        else:
            if to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
                casted_data = to_type.implicit_cast(operand.value, from_type)
            else:
                casted_data = to_type.explicit_cast(operand.value, from_type)
        return Scalar(name=result_scalar.name, data_type=to_type, value=casted_data)

    @classmethod
    def component_evaluation(  # type: ignore[override]
        cls,
        operand: DataComponent,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> DataComponent:
        from_type = operand.data_type
        result_component = cls.component_validation(operand, to_type, mask)
        if mask:
            casted_data = cls.cast_mask_component(operand.data, from_type, to_type, mask)
        else:
            casted_data = cls.cast_component(operand.data, from_type, to_type)

        result_component.data = casted_data
        return result_component
