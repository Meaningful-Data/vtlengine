import pytest

from vtlengine.Model import Scalar, ScalarSet
from vtlengine.Exceptions import InputValidationException
from vtlengine.DataTypes import (
    String,
    Number,
    Integer,
    Date,
    TimePeriod,
    TimeInterval,
    Duration,
    Boolean,
)


@pytest.mark.parametrize(
    "data_type,value",
    [
        # String
        (String, "a"),
        (String, 1),
        (String, 3.14),
        (Number, .1),
        (Number, 1.),
        (String, True),
        (String, False),
        (String, None),
        # Number
        (Number, 3.14),
        (Number, 1),
        (Number, .1),
        (Number, 1.),
        (Number, True),
        (Number, False),
        (Number, "3.14"),
        (Number, "1"),
        (Number, ".1"),
        (Number, "1."),
        (Number, "true"),
        (Number, "FALSE"),
        (Number, "True"),
        (Number, "FaLsE"),
        (Number, None),
        # Integer
        (Integer, 3),
        (Integer, 3.0),
        (Integer, True),
        (Integer, False),
        (Integer, "42"),
        (Integer, "true"),
        (Integer, "FALSE"),
        (Integer, "True"),
        (Integer, "FaLsE"),
        (Integer, None),
        # Boolean
        (Boolean, True),
        (Boolean, False),
        (Boolean, 1),
        (Boolean, 0.0),
        (Boolean, "true"),
        (Boolean, "FALSE"),
        (Boolean, "True"),
        (Boolean, "FaLsE"),
        (Boolean, None),
        # Date
        (Date, "2020-02-29"),
        (Date, "2021-01-01"),
        (Date, None),
        # TimeInterval
        (TimeInterval, "2020"),
        (TimeInterval, "2020-02"),
        (TimeInterval, "2020-01-01/2020-12-31"),
        (TimeInterval, None),
        # TimePeriod
        (TimePeriod, "2020"),
        (TimePeriod, "2020-02"),
        (TimePeriod, "2020-Q1"),
        (TimePeriod, None),
        # Duration
        (Duration, "P3Y2M10D"),
        (Duration, "P0D"),
        (Duration, "P1Y"),
        (Duration, None),
    ],
)
def test_scalar_valid_values(data_type, value):
    s = Scalar(name="test", data_type=data_type, value=value)
    assert s.value == value


@pytest.mark.parametrize(
    "data_type,value",
    [
        # Number
        (Number, "a"),
        # Integer
        (Integer, "3.0"),
        (Integer, 3.14),
        (Integer, "3.14"),
        (Integer, "a"),
        # Boolean
        (Boolean, "a"),
        (Boolean, "yes"),
        # Date
        (Date, "2020-13-01"),
        (Date, "2020-02-30"),
        (Date, "1799-12-31"),
        (Date, "10000-01-01"),
        (Date, True),
        # TimeInterval
        (TimeInterval, "2020-12-31/2020-01-01"),
        (TimeInterval, 1),
        (TimeInterval, "AAAA"),
        (TimeInterval, False),
        # TimePeriod
        (TimePeriod, "2020-13"),
        (TimePeriod, 1),
        (TimePeriod, "AAAA"),
        (TimePeriod, True),
        # Duration
        (Duration, "PX"),
        (Duration, 1),
        (Duration, "AAAA"),
        (Duration, False),
    ],
)
def test_scalar_invalid_values(data_type, value):
    with pytest.raises(InputValidationException):
        Scalar(name="test", data_type=data_type, value=value)


@pytest.mark.parametrize(
    "data_type,values",
    [
        # Integer
        (Integer, [1, 2, 3]),
        # Number
        (Number, [1.5, "2", ".5"]),
        # Boolean
        (Boolean, [True, False, 0, 1, "0", "1", "true", "false"]),
        # String
        (String, ["a", 1, None]),
        # Date
        (Date, ["2020-01-01", "2021-12-31"]),
        # TimeInterval
        (TimeInterval, ["2020-01-01/2020-12-31", "2020", "2020-02"]),
        # TimePeriod
        (TimePeriod, ["2020", "2020-02", "2020-Q4"]),
        # Duration
        (Duration, ["P1Y", "P2M", "P3D", "P1Y2M3D"]),
    ],
)
def test_scalarset_valid_values(data_type, values):
    ss = ScalarSet(data_type=data_type, values=values)
    assert ss.values == values


@pytest.mark.parametrize(
    "data_type,values",
    [
        # Integer
        (Integer, [1, "3.14"]),
        (Integer, [1, 2.5]),
        # Number
        (Number, ["a"]),
        # Boolean
        (Boolean, ["yes"]),
        # Date
        (Date, ["2020-02-30"]),
        # TimeInterval
        (TimeInterval, ["2020-12-31/2020-01-01"]),
        # TimePeriod
        (TimePeriod, ["2020-13"]),
        # Duration
        (Duration, ["PX", "P1Y2X"]),
    ],
)
def test_scalarset_invalid_values(data_type, values):
    with pytest.raises(InputValidationException):
        ScalarSet(data_type=data_type, values=values)

