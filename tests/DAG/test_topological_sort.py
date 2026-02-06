"""Tests for memory-optimized topological sort ordering."""

import networkx as nx
import pytest

from vtlengine.AST.DAG import DAGAnalyzer
from vtlengine.AST.DAG._models import StatementDeps
from vtlengine.Exceptions import SemanticError


def _build_dag_and_sort(vertices: dict, edges: list) -> list:
    """Helper: build a DAGAnalyzer with given vertices/edges and return sort order."""
    dag = DAGAnalyzer()
    dag.vertex = dict(vertices)
    dag.edges = dict(enumerate(edges))
    # Populate dependencies so cycle detection can access them
    dag.dependencies = {k: StatementDeps() for k in vertices}
    dag._build_and_sort_graph("test")
    return dag.sorting


def _is_valid_topological_order(sorting: list, edges: list) -> bool:
    """Check that every edge (u, v) has u before v in the sorting."""
    pos = {node: i for i, node in enumerate(sorting)}
    return all(pos[u] < pos[v] for u, v in edges)


def _peak_live_set(sorting: list, edges: list) -> int:
    """Compute the peak number of intermediate datasets alive at any point.

    Only counts nodes with successors (producers/intermediates), not leaf nodes
    (final outputs). A producer is 'alive' from when it's scheduled until its
    last consumer is scheduled.
    """
    G = nx.DiGraph()
    G.add_nodes_from(sorting)
    G.add_edges_from(edges)
    remaining_consumers = {n: G.out_degree(n) for n in sorting}
    intermediates = {n for n in sorting if G.out_degree(n) > 0}
    alive = set()
    peak = 0
    for node in sorting:
        if node in intermediates:
            alive.add(node)
        for pred in G.predecessors(node):
            remaining_consumers[pred] -= 1
            if remaining_consumers[pred] == 0:
                alive.discard(pred)
        peak = max(peak, len(alive))
    return peak


class TestMemoryOptimalSort:
    def test_linear_chain(self):
        vertices = {1: "a", 2: "b", 3: "c"}
        edges = [(1, 2), (2, 3)]
        result = _build_dag_and_sort(vertices, edges)
        assert result == [1, 2, 3]

    def test_diamond(self):
        vertices = {1: "a", 2: "b", 3: "c"}
        edges = [(1, 3), (2, 3)]
        result = _build_dag_and_sort(vertices, edges)
        assert _is_valid_topological_order(result, edges)
        assert result == [1, 2, 3]

    def test_independent_branches_interleaved(self):
        """Two independent chains: a->b, c->d.
        Should complete one branch before starting the next. Peak = 1.
        """
        vertices = {1: "a", 2: "b", 3: "c", 4: "d"}
        edges = [(1, 2), (3, 4)]
        result = _build_dag_and_sort(vertices, edges)
        assert _is_valid_topological_order(result, edges)
        assert result.index(2) == result.index(1) + 1
        assert _peak_live_set(result, edges) == 1

    def test_fan_in_fan_out_pattern(self):
        """Simplified test_bdi: two pairs of producers, each feeding four consumers.
        Peak should be 2 (one pair at a time), not 4 (all producers at once).
        """
        vertices = {
            1: "DS_1",
            2: "DS_2",
            3: "chk_1",
            4: "chk_2",
            5: "chk_3",
            6: "chk_4",
            7: "DS_3",
            8: "DS_4",
            9: "chk_5",
            10: "chk_6",
            11: "chk_7",
            12: "chk_8",
        }
        edges = [
            (1, 3),
            (2, 3),
            (1, 4),
            (2, 4),
            (1, 5),
            (2, 5),
            (1, 6),
            (2, 6),
            (7, 9),
            (8, 9),
            (7, 10),
            (8, 10),
            (7, 11),
            (8, 11),
            (7, 12),
            (8, 12),
        ]
        result = _build_dag_and_sort(vertices, edges)
        assert _is_valid_topological_order(result, edges)
        assert _peak_live_set(result, edges) <= 2

    def test_single_node(self):
        vertices = {1: "a"}
        result = _build_dag_and_sort(vertices, [])
        assert result == [1]

    def test_no_edges_deterministic(self):
        vertices = {3: "c", 1: "a", 2: "b"}
        result = _build_dag_and_sort(vertices, [])
        assert result == [1, 2, 3]

    def test_cycle_detection(self):
        vertices = {1: "a", 2: "b"}
        edges = [(1, 2), (2, 1)]
        with pytest.raises(SemanticError):
            _build_dag_and_sort(vertices, edges)

    def test_wide_fan_in(self):
        vertices = {1: "a", 2: "b", 3: "c", 4: "sink"}
        edges = [(1, 4), (2, 4), (3, 4)]
        result = _build_dag_and_sort(vertices, edges)
        assert _is_valid_topological_order(result, edges)
        assert result[-1] == 4
