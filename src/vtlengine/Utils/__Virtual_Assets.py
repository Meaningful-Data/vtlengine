import threading
from copy import copy

# The counts of the virtual names, for the thread that asks for them: runs in concurrent threads
# number their virtual Data Sets and Components on their own (issue #1352)
_counts = threading.local()


class _ThreadCounts(type):
    @property
    def dataset_count(cls) -> int:
        return int(getattr(_counts, "dataset_count", 0))

    @dataset_count.setter
    def dataset_count(cls, value: int) -> None:
        _counts.dataset_count = value

    @property
    def component_count(cls) -> int:
        return int(getattr(_counts, "component_count", 0))

    @component_count.setter
    def component_count(cls, value: int) -> None:
        _counts.component_count = value


class VirtualCounter(metaclass=_ThreadCounts):
    _instance = None

    def __init__(self) -> None:
        self.dataset_count = 0
        self.component_count = 0

    def __new__(cls):  # type: ignore[no-untyped-def]
        if cls._instance is None:
            cls._instance = super(VirtualCounter, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls.dataset_count = 0
        cls.component_count = 0

    @classmethod
    def _new_ds_name(cls) -> str:
        cls.dataset_count += 1
        name = f"__VDS_{copy(cls.dataset_count)}__"
        return name

    @classmethod
    def _new_dc_name(cls) -> str:
        cls.component_count += 1
        name = f"__VDC_{copy(cls.component_count)}__"
        return name
