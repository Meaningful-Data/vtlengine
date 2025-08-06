from copy import copy


class VirtualCounter:
    _instance = None
    dataset_count: int = 0
    component_count: int = 0

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
