"""
Large Script Tests

Scripts generated from validation rule sets chain many datasets in a single expression.
These tests pin how the DuckDB backend transpiles them.

Naming conventions:
- Identifiers: Id_1, Id_2, etc.
- Measures: Me_1, Me_2, etc.
"""

from typing import Dict, List, Sequence

import pytest

from vtlengine.AST import Assignment, BinOp, Start, VarID
from vtlengine.DataTypes import Number, String
from vtlengine.duckdb_transpiler.Transpiler import SQLTranspiler
from vtlengine.Model import Component, Dataset, Role

# =============================================================================
# Helpers
# =============================================================================


def _node(**kwargs: object) -> Dict[str, object]:
    return {"line_start": 1, "column_start": 1, "line_stop": 1, "column_stop": 10, **kwargs}


def _dataset(
    name: str,
    ids: Sequence[str] = ("Id_1",),
    measures: Sequence[str] = ("Me_1",),
    measure_type: type = Number,
    id_type: type = String,
    viral: Sequence[str] = (),
) -> Dataset:
    comps = {
        i: Component(name=i, data_type=id_type, role=Role.IDENTIFIER, nullable=False) for i in ids
    }
    for m in measures:
        comps[m] = Component(name=m, data_type=measure_type, role=Role.MEASURE, nullable=True)
    for v in viral:
        comps[v] = Component(name=v, data_type=String, role=Role.VIRAL_ATTRIBUTE, nullable=True)
    return Dataset(name=name, components=comps, data=None)


def _chain(names: List[str], ops: List[str]) -> BinOp:
    """Left-nested ``names[0] ops[0] names[1] ops[1] ...``."""
    expr = VarID(**_node(value=names[0]))
    for name, op in zip(names[1:], ops):
        expr = BinOp(**_node(left=expr, op=op, right=VarID(**_node(value=name))))
    return expr


def _transpile(datasets: Dict[str, Dataset], expr: BinOp, output: Dataset) -> str:
    transpiler = SQLTranspiler(
        input_datasets=datasets,
        output_datasets={"DS_r": output},
        input_scalars={},
        output_scalars={},
    )
    assignment = Assignment(**_node(left=VarID(**_node(value="DS_r")), op=":=", right=expr))
    return transpiler.transpile(Start(**_node(children=[assignment])))[0][1]


# =============================================================================
# Chains of datasets
# =============================================================================


class TestLongChainTranspile:
    """Transpiling a long chain looks each structure up once."""

    def test_long_chain_resolves_each_structure_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each level of a chain resolves its operands again, which the memo absorbs."""
        names = [f"DS_{i}" for i in range(60)]
        datasets = {name: _dataset(name) for name in names}
        resolved = []
        original = SQLTranspiler._resolve_dataset_structure

        def counting(self: SQLTranspiler, node: object) -> object:
            resolved.append(node)
            return original(self, node)

        monkeypatch.setattr(SQLTranspiler, "_resolve_dataset_structure", counting)
        # ``*`` keeps the pairwise joins, where the structures are looked up level by level
        sql = _transpile(datasets, _chain(names, ["*"] * 59), datasets["DS_0"])

        assert sql.count(" JOIN ") == 59
        assert len(resolved) == len({id(node) for node in resolved})
