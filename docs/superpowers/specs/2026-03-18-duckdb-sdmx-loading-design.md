# DuckDB SDMX Loading — Design Spec

**Issue:** [#475 — (Duckdb) Implement SDMX loading](https://github.com/Meaningful-Data/vtlengine/issues/475)
**Date:** 2026-03-18

## Context

The Pandas backend supports full SDMX data loading: `run_sdmx()` with `PandasDataset` objects, SDMX-ML/JSON file loading via pysdmx, and URL-based SDMX datapoint fetching. The DuckDB backend currently only supports CSV files and direct DataFrames. This creates a feature gap — users who load data via SDMX cannot use the DuckDB execution engine.

**Goal:** Achieve full SDMX loading parity in the DuckDB backend by routing SDMX data through pysdmx → DataFrame → DuckDB table.

## Approach

**Thin pass-through:** Extend existing functions to handle SDMX inputs, converting them to DataFrames via pysdmx, then registering as DuckDB tables using the existing `register_dataframes()` path. No new modules or abstractions needed.

## Data Flow

```
PandasDataset ──→ run_sdmx(use_duckdb=True)
                      │
                      ├─ Schema → to_vtl_json() → data_structures
                      └─ .data → Dict[str, DataFrame] → datapoints
                                        │
                                        ▼
                              run(use_duckdb=True)
                                        │
                                        ▼
                              _run_with_duckdb()
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              CSV files           DataFrames          SDMX files/URLs
                    │                   │                   │
          load_datapoints_duckdb  register_dataframes  pysdmx → DataFrame
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                              DuckDB table + validation
```

## Changes

### 1. API Layer (`src/vtlengine/API/__init__.py`)

**`run_sdmx()` — add `use_duckdb` parameter:**
- New parameter: `use_duckdb: bool = False`
- Pass it through to the `run()` call at line ~713
- No other changes needed — `run_sdmx()` already converts PandasDatasets to DataFrames + VTL JSON structures

**`_run_with_duckdb()` — add `sdmx_mappings` parameter:**
- New parameter: `sdmx_mappings: Optional[Union[VtlDataflowMapping, Dict[str, str]]] = None`
- Convert to dict via `_convert_sdmx_mappings()` at the start of the function
- **Exact call site (line 324):** Change `load_datasets(data_structures)` to `load_datasets(data_structures, sdmx_mappings=mapping_dict)` — this ensures pysdmx structure objects (Schema, DSD, Dataflow) resolve correctly when passed as `data_structures`

**`_run_with_duckdb()` — handle URL datapoints before `extract_datapoint_paths()`:**
- Before calling `extract_datapoint_paths()`, detect URL values in `datapoints` dict
- Load URL datapoints via `_handle_url_datapoints(url_datapoints, data_structures, mapping_dict)` from `_InternalApi.py`
- `_handle_url_datapoints()` returns `(datasets, scalars, dataframes)` — merge the returned `datasets` into `input_datasets` (the returned datasets contain schema info derived from the SDMX response)
- Merge the returned `dataframes` into the `datapoints` dict as DataFrames
- Remove URL entries from `datapoints` dict so `extract_datapoint_paths()` only sees files and DataFrames

```python
# In _run_with_duckdb(), before extract_datapoint_paths():
if isinstance(datapoints, dict):
    url_datapoints = {k: v for k, v in datapoints.items()
                      if isinstance(v, str) and _is_url(v)}
    if url_datapoints:
        url_ds, _, url_dfs = _handle_url_datapoints(
            url_datapoints, data_structures, mapping_dict
        )
        input_datasets.update(url_ds)
        for name, df in url_dfs.items():
            datapoints[name] = df
        for name in url_datapoints:
            if name in datapoints and isinstance(datapoints[name], str):
                del datapoints[name]
```

**`run()` — pass `sdmx_mappings` to `_run_with_duckdb()`:**
- Forward the existing `sdmx_mappings` parameter to `_run_with_duckdb()` when `use_duckdb=True` (line ~529)

### 2. DuckDB IO Layer (`src/vtlengine/duckdb_transpiler/io/_io.py`)

**`extract_datapoint_paths()` — recognize SDMX files:**

Currently handles:
- `Dict[str, DataFrame]` → routes to `df_dict`
- `Dict[str, Path/str]` → routes to `path_dict`
- `List[Path/str]` → routes to `path_dict`
- Single `Path/str` → routes to `path_dict`

New behavior for file paths:
- When a path has `.xml` or `.json` extension, check if it's an SDMX datapoint file using `is_sdmx_datapoint_file()` from `vtlengine/files/sdmx_handler.py` (defined there at line 36, re-exported from `parser/__init__.py`)
- If SDMX: load via `load_sdmx_datapoints(components, dataset_name, file_path)` from `vtlengine/files/sdmx_handler.py` → route resulting DataFrame to `df_dict`
- If not SDMX (plain CSV): route to `path_dict` as before
- **For `.json` files:** If `load_sdmx_datapoints()` raises an exception, fall back to treating it as a regular file path (route to `path_dict`). This handles ambiguity between SDMX-JSON and plain JSON files.

**Name resolution strategy for SDMX files:**
- **Dict inputs** (e.g. `{"DS_1": Path("file.xml")}`): Use the dict key as `dataset_name`, get components from `input_datasets[name].components`
- **List or single Path inputs** (e.g. `[Path("file.xml")]`): Call `extract_sdmx_dataset_name(path)` from `vtlengine/files/sdmx_handler.py` to get the dataset name from the file content, then get components from `input_datasets[name].components`

**Note:** URL handling is done at the `_run_with_duckdb()` level (see Section 1), so `extract_datapoint_paths()` does not need to handle URLs.

### 3. Validation in `register_dataframes()` (`src/vtlengine/duckdb_transpiler/io/_io.py`)

Add the same post-load validation that `load_datapoints_duckdb()` performs for CSV files. Include error-handling cleanup (drop table on validation failure) to match the pattern in `load_datapoints_duckdb()` (lines 225-227):

```python
if not SKIP_LOAD_VALIDATION:
    try:
        id_columns = [n for n, c in components.items() if c.role == Role.IDENTIFIER]
        # DWI: no identifiers → max 1 row
        if not id_columns:
            result = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()
            if result and result[0] > 1:
                raise DataLoadError("0-3-1-4", name=name)
        # Duplicate check
        validate_no_duplicates(conn, name, id_columns)
        # Temporal type validation
        validate_temporal_columns(conn, name, components)
    except DataLoadError:
        conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        raise
```

**Fix column ordering in INSERT:** The current `INSERT INTO ... SELECT * FROM` assumes DataFrame column order matches table schema. Change to use explicit column names to prevent silent data misplacement with SDMX-derived DataFrames:

```python
col_list = ", ".join(f'"{c}"' for c in components)
conn.execute(f'INSERT INTO "{name}" ({col_list}) SELECT {col_list} FROM "{temp_view}"')
```

### 4. DuckDB IO Exports (`src/vtlengine/duckdb_transpiler/io/__init__.py`)

No changes needed — `extract_datapoint_paths` and `register_dataframes` are already exported.

### 5. Testing

**Extend `tests/API/test_sdmx.py`:**

Add DuckDB variants of existing SDMX tests. These follow the pattern in `tests/Additional/test_additional_scalars.py` which calls `run()` with `use_duckdb=_use_duckdb_backend()`:

Happy path tests:
- `test_run_sdmx_function_duckdb()` — `run_sdmx(use_duckdb=True)` with PandasDataset objects
- `test_run_sdmx_function_with_mappings_duckdb()` — with VtlDataflowMapping and dict mappings
- `test_run_sdmx_file_via_dict_duckdb()` — SDMX-ML file in dict, `run(use_duckdb=True)`
- `test_run_sdmx_file_via_list_duckdb()` — SDMX-ML file in list, `run(use_duckdb=True)`
- `test_run_with_url_datapoints_duckdb()` — URL datapoints with mocked pysdmx, `run(use_duckdb=True)`
- `test_run_with_schema_object_duckdb()` — pysdmx Schema as data_structures, `run(use_duckdb=True)`
- `test_run_with_dsd_object_duckdb()` — DataStructureDefinition, `run(use_duckdb=True)`

Error/negative tests:
- `test_run_sdmx_invalid_structure_duckdb()` — PandasDataset with non-Schema structure raises `InputValidationException`
- `test_run_sdmx_file_duplicate_ids_duckdb()` — SDMX file with duplicate identifiers raises `DataLoadError`
- `test_run_sdmx_url_without_structure_duckdb()` — URL datapoints without valid structure reference raises error

Reuse existing test fixtures (sdmx_data_file, sdmx_structure_file, sdmx_data_structure) and reference data.

**Extend `tests/duckdb_transpiler/test_efficient_io.py`:**

- `test_register_dataframes_validates_duplicates()` — verify duplicate check fires
- `test_register_dataframes_validates_temporal_types()` — verify temporal validation fires
- `test_register_dataframes_drops_table_on_validation_failure()` — verify cleanup on error
- `test_extract_datapoint_paths_sdmx_file()` — verify SDMX files route to df_dict

## Files to Modify

| File | Change |
|------|--------|
| `src/vtlengine/API/__init__.py` | Add `use_duckdb` to `run_sdmx()`, `sdmx_mappings` + URL handling to `_run_with_duckdb()`, forward mappings in `run()` |
| `src/vtlengine/duckdb_transpiler/io/_io.py` | Extend `extract_datapoint_paths()` for SDMX files, add validation + column-safe INSERT to `register_dataframes()` |
| `tests/API/test_sdmx.py` | Add DuckDB variants of SDMX tests (happy path + error cases) |
| `tests/duckdb_transpiler/test_efficient_io.py` | Add validation, cleanup, and SDMX detection tests |

## Existing Functions to Reuse

| Function | Location | Purpose |
|----------|----------|---------|
| `is_sdmx_datapoint_file()` | `src/vtlengine/files/sdmx_handler.py` (line 36) | Detect SDMX file format |
| `load_sdmx_datapoints()` | `src/vtlengine/files/sdmx_handler.py` | Load SDMX-ML/JSON → DataFrame |
| `extract_sdmx_dataset_name()` | `src/vtlengine/files/sdmx_handler.py` | Extract dataset name from SDMX file |
| `to_vtl_json()` | `src/vtlengine/files/sdmx_handler.py` | Convert Schema → VTL JSON |
| `_handle_url_datapoints()` | `src/vtlengine/API/_InternalApi.py` | Fetch SDMX data from URLs |
| `_is_url()` | `src/vtlengine/API/_InternalApi.py` | Detect HTTP/HTTPS URLs |
| `_convert_sdmx_mappings()` | `src/vtlengine/API/_sdmx_utils.py` | Convert VtlDataflowMapping → dict |
| `validate_no_duplicates()` | `src/vtlengine/duckdb_transpiler/io/_validation.py` | Post-load duplicate check |
| `validate_temporal_columns()` | `src/vtlengine/duckdb_transpiler/io/_validation.py` | Post-load temporal type check |
| `build_create_table_sql()` | `src/vtlengine/duckdb_transpiler/io/_validation.py` | Generate CREATE TABLE SQL |

## Verification

1. **Unit tests:** Run `poetry run pytest tests/API/test_sdmx.py -v` and `poetry run pytest tests/duckdb_transpiler/test_efficient_io.py -v`
2. **Integration test:** Run a `run_sdmx(use_duckdb=True)` call manually with a PandasDataset:
   ```python
   from pysdmx.io import get_datasets
   from vtlengine import run_sdmx
   datasets = get_datasets(data="path/to/sdmx_data.xml", structure="path/to/structure.xml")
   result = run_sdmx(script="DS_r <- DS_1;", datasets=datasets, use_duckdb=True)
   ```
3. **Full test suite:** `poetry run pytest` with `VTL_ENGINE_BACKEND=duckdb`
4. **Code quality:** `poetry run ruff format && poetry run ruff check --fix --unsafe-fixes && poetry run mypy`
