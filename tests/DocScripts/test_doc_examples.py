"""Test that Python code examples in RST documentation files run correctly."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pytest

from tests.DocScripts._rst_code_extractor import CodeBlock, extract_python_blocks, is_runnable
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Dataset, Scalar

docs_dir = Path(__file__).resolve().parents[2] / "docs"
static_dir = docs_dir / "_static"


def get_runnable_blocks() -> List[Tuple[str, CodeBlock]]:
    """Scan all RST files in docs/ and return runnable Python code blocks."""
    blocks: List[Tuple[str, CodeBlock]] = []
    for rst_file in sorted(docs_dir.glob("*.rst")):
        for block in extract_python_blocks(rst_file):
            if is_runnable(block):
                blocks.append((rst_file.name, block))
    return blocks


def _block_id(item: Tuple[str, CodeBlock]) -> str:
    rst_name, block = item
    return f"{rst_name}::line_{block.line_number}"


def _preprocess_for_result_capture(source: str) -> str:
    """Transform source code to capture run results in the namespace.

    Handles two patterns:
    1. def main() blocks that print results — replace print with return, add main() call.
    2. Inline print(run(...)[...]) calls — replace with a plain assignment.
    """
    if "def main():" in source:
        source = re.sub(r"\bprint\(run_result\)", "return run_result", source)
        source = re.sub(r"\bprint\(result\)", "return result", source)
        source += "\nrun_result = main()\n"
        return source

    # Replace print(run_sdmx(...)[...].data) or print(run(...)[...]) with assignment
    source = re.sub(
        r"\bprint\((run(?:_sdmx)?\([^)]*\)).*\)",
        r"run_result = \1",
        source,
    )
    return source


def _exec_block(source: str, filename: str, capture_results: bool = False) -> dict[str, object]:
    """Execute a code block and return the resulting namespace."""
    if capture_results:
        source = _preprocess_for_result_capture(source)
    namespace: dict[str, object] = {}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace


def _find_result_datasets(
    namespace: dict[str, object],
) -> Dict[str, Union[Dataset, Scalar]]:
    """Extract run result datasets from the exec namespace."""
    for var_name in ("run_result", "result"):
        val = namespace.get(var_name)
        if isinstance(val, dict) and any(isinstance(v, (Dataset, Scalar)) for v in val.values()):
            return val  # type: ignore[return-value]

    # Fallback: any dict of Datasets
    for val in namespace.values():
        if isinstance(val, dict) and any(isinstance(v, (Dataset, Scalar)) for v in val.values()):
            return val  # type: ignore[return-value]
    return {}


def _validate_csv_outputs(
    namespace: dict[str, object],
    csv_references: List[str],
) -> None:
    """Compare run results against reference CSV files."""
    results = _find_result_datasets(namespace)
    for csv_file in csv_references:
        csv_path = static_dir / csv_file
        expected_df = pd.read_csv(csv_path, dtype_backend="pyarrow")

        dataset_name = _csv_to_dataset_name(csv_file, results)
        assert dataset_name is not None, (
            f"Could not match {csv_file} to any result dataset. Available: {list(results.keys())}"
        )

        result = results[dataset_name]
        if isinstance(result, Scalar):
            continue
        assert isinstance(result, Dataset)
        actual_df = result.data.reset_index(drop=True)
        expected_df = expected_df.reset_index(drop=True)

        # Normalize empty strings to NA and cast actual to match expected types
        actual_df = actual_df.replace("", pd.NA)
        for col in expected_df.columns:
            if col in actual_df.columns:
                actual_df[col] = actual_df[col].astype(expected_df[col].dtype)

        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)


def _csv_to_dataset_name(
    csv_file: str,
    results: Dict[str, Union[Dataset, Scalar]],
) -> Optional[str]:
    """Map a CSV filename to a dataset name in the results dict."""
    stem = csv_file.removesuffix(".csv")
    # Direct match
    if stem in results:
        return stem
    # Strip common suffixes
    for suffix in ("_run", "_run_sdmx", "_run_with_scalars", "_output", "_2_output"):
        candidate = stem.removesuffix(suffix)
        if candidate != stem and candidate in results:
            return candidate
    return None


runnable_blocks = get_runnable_blocks()


@pytest.mark.parametrize(
    "rst_file,block",
    runnable_blocks,
    ids=[_block_id(b) for b in runnable_blocks],
)
def test_doc_example(rst_file: str, block: CodeBlock) -> None:
    """Execute a Python code block extracted from documentation."""
    filename = f"<{rst_file}:{block.line_number}>"
    if block.expects_error:
        with pytest.raises(SemanticError):
            _exec_block(block.source, filename)
    else:
        namespace = _exec_block(block.source, filename, capture_results=bool(block.csv_references))
        if block.csv_references:
            _validate_csv_outputs(namespace, block.csv_references)
