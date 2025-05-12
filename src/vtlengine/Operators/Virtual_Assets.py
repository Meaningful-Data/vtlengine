class VirtualCounter:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VirtualCounter, cls).__new__(cls)
            cls._instance._reset()
        return cls._instance

    def _reset(self):
        self.dataset_count = 0
        self.component_count = 0

    def _new_ds_name(self) -> str:
        name = f"@VDS_{self.dataset_count}"
        self.dataset_count += 1
        return name

    def _new_dc_name(self) -> str:
        name = f"@VDC_{self.component_count}"
        self.component_count += 1
        return name

    def reset(self):
        self._reset()
