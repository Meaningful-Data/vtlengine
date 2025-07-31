import uuid
from copy import copy
from typing import List

from vtlengine.connection import con


class VirtualCounter:
    _instance = None
    dataset_count: int = 0
    component_count: int = 0
    temp_views: List[str] = []

    def __init__(self) -> None:
        self.dataset_count = 0
        self.component_count = 0
        self.temp_views = []

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
    def reset_temp_views(cls) -> None:
        for view in cls.temp_views:
            try:
                con.unregister(view)
            except Exception as e:
                print(f"Error dropping view {view}: {e}")
        cls.temp_views = []

    # TODO: DuckDB have problem operating columns with @ in their names
    #  Virtual name @ have been replaced with __VDS_...__ and __VDC_...__ to avoid this issue
    #  until we consider the names changes.
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

    @classmethod
    def _new_temp_view_name(cls) -> str:
        name = f"__TMP_{uuid.uuid4().hex[:8]}__"
        cls.temp_views.append(name)
        return name
