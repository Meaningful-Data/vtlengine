import pytest

from vtlengine.DataTypes.TimeHandling import TimePeriodConfig


@pytest.fixture(autouse=True)
def _reset_time_period_config() -> None:  # type: ignore[misc]
    """Reset TimePeriodConfig to default after every test to prevent state leakage."""
    yield  # type: ignore[misc]
    TimePeriodConfig.set_representation("vtl")
