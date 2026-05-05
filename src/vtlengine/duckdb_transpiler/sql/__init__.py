"""SQL initialization for VTL time types in DuckDB."""

import re
import weakref
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, Iterator, List, Optional, Set

if TYPE_CHECKING:
    import duckdb

_SQL_DIR = Path(__file__).parent
_SQL_FILES = (_SQL_DIR / "init.sql", _SQL_DIR / "time_operators.sql")

# WeakSet so closed connections are pruned automatically; used to skip work
# on idempotent re-installs of the full library.
_initialized_connections: "weakref.WeakSet[duckdb.DuckDBPyConnection]" = weakref.WeakSet()

_CREATE_HEADER = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:MACRO|TYPE)\s+([A-Za-z_]\w*)",
    re.IGNORECASE,
)
_DROP_HEADER = re.compile(r"^\s*DROP\s+TYPE\s+IF\s+EXISTS\s+([A-Za-z_]\w*)", re.IGNORECASE)
_VTL_REF = re.compile(r"\bvtl_[a-z_][a-z0-9_]*\b")
_LINE_COMMENT = re.compile(r"--[^\n]*")


@dataclass(frozen=True)
class _MacroGraph:
    """Parsed view of the SQL library: each named object plus its deps."""

    statements: Dict[str, str]
    deps: Dict[str, FrozenSet[str]]
    order: tuple  # type: ignore[type-arg]


@lru_cache(maxsize=1)
def _read_full_sql() -> str:
    """Read the SQL library files concatenated (cached for the process)."""
    return "\n".join(p.read_text() for p in _SQL_FILES if p.exists())


def _iter_statements(sql: str) -> Iterator[str]:
    """Yield non-empty top-level SQL statements."""
    for raw in sql.split(";"):
        stmt = raw.strip()
        if stmt:
            yield stmt


@lru_cache(maxsize=1)
def _macro_graph() -> _MacroGraph:
    """Parse the SQL library into a ``_MacroGraph``."""
    statements: Dict[str, str] = {}
    deps: Dict[str, FrozenSet[str]] = {}
    order: List[str] = []
    pending_drops: Dict[str, str] = {}

    for stmt in _iter_statements(_read_full_sql()):
        head = _LINE_COMMENT.sub("", stmt).lstrip()

        drop_match = _DROP_HEADER.match(head)
        if drop_match:
            pending_drops[drop_match.group(1)] = stmt + ";"
            continue

        create_match = _CREATE_HEADER.match(head)
        if not create_match:
            continue

        name = create_match.group(1)
        # Strip line comments before scanning for refs so commented-out names
        # don't create phantom dependencies.
        body = _LINE_COMMENT.sub("", stmt[create_match.end() :])
        refs = frozenset(ref for ref in _VTL_REF.findall(body) if ref != name)

        prefix = pending_drops.pop(name, "")
        statements[name] = (prefix + " " + stmt + ";").lstrip()
        deps[name] = refs
        order.append(name)

    return _MacroGraph(statements=statements, deps=deps, order=tuple(order))


def _closure(seeds: Iterable[str], deps: Dict[str, FrozenSet[str]]) -> Set[str]:
    """Return the transitive closure of ``seeds`` over ``deps``."""
    needed: Set[str] = set()
    stack = [s for s in seeds if s in deps]
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        stack.extend(deps[name] - needed)
    return needed


def _required_macros_sql(sql_fragments: Iterable[str]) -> Optional[str]:
    """Return the minimal SQL needed for ``sql_fragments``, or ``None`` if no
    VTL macros are referenced."""
    graph = _macro_graph()
    seeds = {ref for frag in sql_fragments for ref in _VTL_REF.findall(frag)}
    seeds &= graph.statements.keys()
    if not seeds:
        return None
    needed = _closure(seeds, graph.deps)
    return "\n".join(graph.statements[name] for name in graph.order if name in needed)


def initialize_time_types(
    conn: "duckdb.DuckDBPyConnection",
    sql_fragments: Optional[Iterable[str]] = None,
) -> None:
    """Install VTL time types and macros on ``conn``."""
    if conn in _initialized_connections:
        return

    if sql_fragments is None:
        conn.execute(_read_full_sql())
        _initialized_connections.add(conn)
        return

    minimal_sql = _required_macros_sql(sql_fragments)
    if minimal_sql is not None:
        conn.execute(minimal_sql)
