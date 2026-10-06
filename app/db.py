"""SQLite 持久层。

版本模型
--------
每次导入（步道图或封闭记录）生成一个新版本号：

- 新版本完整复制上一版本的节点/路段/封闭记录，再覆盖被更新的那部分；
- ``meta`` 表中的 ``current_version`` 指针在同一事务内切换；
- 查询通过 :func:`get_snapshot` 在开始时读取一次指针，整次请求固定使用该版本。

导入失败时事务整体回滚，旧版本保持不动。

版本恢复（:func:`restore_version`）把指定历史版本的四类数据整体复制为一个
**新的**生效版本：版本号继续递增、历史版本原样保留，绝不回拨指针。恢复与
导入共用同一 ``BEGIN IMMEDIATE`` 写事务，彼此串行，生效顺序即提交顺序。
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Optional

from . import config
from .schemas import ValidationError, normalize_closures, normalize_graph

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
    version_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL CHECK (kind IN ('graph', 'closures', 'restore')),
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS nodes (
    version_id  INTEGER NOT NULL,
    node_id     TEXT NOT NULL,
    PRIMARY KEY (version_id, node_id)
);

CREATE TABLE IF NOT EXISTS edges (
    version_id      INTEGER NOT NULL,
    edge_id         TEXT NOT NULL,
    from_node       TEXT NOT NULL,
    to_node         TEXT NOT NULL,
    travel_seconds  INTEGER NOT NULL CHECK (travel_seconds > 0),
    has_stairs      INTEGER NOT NULL CHECK (has_stairs IN (0, 1)),
    PRIMARY KEY (version_id, edge_id)
);

CREATE TABLE IF NOT EXISTS closures (
    version_id  INTEGER NOT NULL,
    closure_id  INTEGER NOT NULL,
    edge_id     TEXT NOT NULL,
    start_ts    INTEGER NOT NULL,
    end_ts      INTEGER NOT NULL CHECK (end_ts > start_ts),
    PRIMARY KEY (version_id, closure_id)
);

CREATE TABLE IF NOT EXISTS reverse_windows (
    version_id  INTEGER NOT NULL,
    edge_id     TEXT NOT NULL,
    window_idx  INTEGER NOT NULL,
    start_ts    INTEGER NOT NULL,
    end_ts      INTEGER NOT NULL CHECK (end_ts > start_ts),
    PRIMARY KEY (version_id, edge_id, window_idx)
);

CREATE INDEX IF NOT EXISTS idx_edges_from     ON edges(version_id, from_node);
CREATE INDEX IF NOT EXISTS idx_closures_edge ON closures(version_id, edge_id);
CREATE INDEX IF NOT EXISTS idx_reverse_edge  ON reverse_windows(version_id, edge_id);
"""


class EdgeRow(NamedTuple):
    id: str
    src: str
    dst: str
    seconds: int
    has_stairs: bool


class ClosureRow(NamedTuple):
    edge_id: str
    start: int
    end: int


class ReverseWindowRow(NamedTuple):
    edge_id: str
    start: int
    end: int


class Snapshot(NamedTuple):
    """单次查询使用的不可变数据版本。"""

    version_id: int
    nodes: list[str]
    edges: list[EdgeRow]
    closures: list[ClosureRow]
    reverse_windows: list[ReverseWindowRow] = []

    @property
    def node_set(self) -> set[str]:
        return set(self.nodes)


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """建表并写入初始指针（不存在任何已发布版本时）。"""
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('current_version', '0')"
    )
    _migrate_versions_kind(conn)


def _migrate_versions_kind(conn: sqlite3.Connection) -> None:
    """旧库的 ``versions.kind`` CHECK 不含 ``'restore'`` 时，原地重建该表。

    建表语句中的 CHECK 约束无法就地修改，按 SQLite 惯例
    “建新表 → 复制 → 删旧表 → 改名”迁移；已有版本行原样保留。
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='versions'"
    ).fetchone()
    if row is None or "'restore'" in row["sql"]:
        return
    with transaction(conn):
        conn.execute("DROP TABLE IF EXISTS versions_new")
        conn.execute(
            "CREATE TABLE versions_new ("
            "    version_id  INTEGER PRIMARY KEY AUTOINCREMENT,"
            "    kind        TEXT NOT NULL CHECK (kind IN ('graph', 'closures', 'restore')),"
            "    created_at  INTEGER NOT NULL DEFAULT (strftime('%s','now'))"
            ")"
        )
        conn.execute(
            "INSERT INTO versions_new(version_id, kind, created_at) "
            "SELECT version_id, kind, created_at FROM versions"
        )
        conn.execute("DROP TABLE versions")
        conn.execute("ALTER TABLE versions_new RENAME TO versions")


@contextmanager
def transaction(conn: sqlite3.Connection):
    """立即失败的写事务，避免与其他导入互相等待。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key='current_version'"
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def _copy_version(conn: sqlite3.Connection, old: int, new: int, kind: str) -> None:
    conn.execute("INSERT INTO versions(version_id, kind) VALUES (?, ?)", (new, kind))
    if old:
        conn.execute("INSERT INTO nodes(version_id, node_id) SELECT ?, node_id FROM nodes WHERE version_id=?", (new, old))
        conn.execute(
            "INSERT INTO edges(version_id, edge_id, from_node, to_node, travel_seconds, has_stairs) "
            "SELECT ?, edge_id, from_node, to_node, travel_seconds, has_stairs FROM edges WHERE version_id=?",
            (new, old),
        )
        conn.execute(
            "INSERT INTO closures(version_id, closure_id, edge_id, start_ts, end_ts) "
            "SELECT ?, closure_id, edge_id, start_ts, end_ts FROM closures WHERE version_id=?",
            (new, old),
        )
        conn.execute(
            "INSERT INTO reverse_windows(version_id, edge_id, window_idx, start_ts, end_ts) "
            "SELECT ?, edge_id, window_idx, start_ts, end_ts FROM reverse_windows "
            "WHERE version_id=?",
            (new, old),
        )


def publish_graph(conn: sqlite3.Connection, payload) -> int:
    """原子发布新步道图。

    校验失败抛出 :class:`ValidationError` 并完整回滚；旧封闭记录被保留，
    其中引用了不在新图中路段的记录会自动丢弃。
    """
    graph = normalize_graph(payload)
    new_edge_ids = {e["id"] for e in graph["edges"]}

    with transaction(conn):
        old = _current_version(conn)
        new = int(
            conn.execute("SELECT COALESCE(MAX(version_id), 0) + 1 AS v FROM versions").fetchone()["v"]
        )
        _copy_version(conn, old, new, "graph")

        # 用新版本的图整体替换
        conn.execute("DELETE FROM nodes WHERE version_id=?", (new,))
        conn.execute("DELETE FROM edges WHERE version_id=?", (new,))
        conn.executemany(
            "INSERT INTO nodes(version_id, node_id) VALUES (?, ?)",
            [(new, n) for n in graph["nodes"]],
        )
        conn.executemany(
            "INSERT INTO edges(version_id, edge_id, from_node, to_node, travel_seconds, has_stairs) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (new, e["id"], e["src"], e["dst"], e["seconds"], 1 if e["has_stairs"] else 0)
                for e in graph["edges"]
            ],
        )
        # 反向时间窗随新图整体替换（校验已在事务前完成）
        conn.execute("DELETE FROM reverse_windows WHERE version_id=?", (new,))
        conn.executemany(
            "INSERT INTO reverse_windows(version_id, edge_id, window_idx, start_ts, end_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (new, e["id"], idx, start, end)
                for e in graph["edges"]
                for idx, (start, end) in enumerate(e["reverse_windows"])
            ],
        )
        # 丢弃悬挂封闭记录
        conn.execute("DELETE FROM closures WHERE version_id=? AND edge_id NOT IN (%s)" %
                     (",".join("?" * len(new_edge_ids)) if new_edge_ids else "''"),
                     (new, *sorted(new_edge_ids)) if new_edge_ids else (new,))

        conn.execute(
            "UPDATE meta SET value=? WHERE key='current_version'", (str(new),)
        )
    return new


def publish_closures(conn: sqlite3.Connection, payload) -> int:
    """原子整体替换封闭记录。必须先发布过步道图。"""
    with transaction(conn):
        old = _current_version(conn)
        if not old:
            # 回滚由 contextmanager 完成
            raise ValueError("尚未导入步道图，无法发布封闭记录")
        known_edges = {
            r["edge_id"]
            for r in conn.execute("SELECT edge_id FROM edges WHERE version_id=?", (old,))
        }
        closures = normalize_closures(payload, known_edges)

        new = int(
            conn.execute("SELECT COALESCE(MAX(version_id), 0) + 1 AS v FROM versions").fetchone()["v"]
        )
        _copy_version(conn, old, new, "closures")
        conn.execute("DELETE FROM closures WHERE version_id=?", (new,))
        conn.executemany(
            "INSERT INTO closures(version_id, closure_id, edge_id, start_ts, end_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            [(new, i, c["edge_id"], c["start"], c["end"]) for i, c in enumerate(closures)],
        )
        conn.execute(
            "UPDATE meta SET value=? WHERE key='current_version'", (str(new),)
        )
    return new


class VersionNotFoundError(ValueError):
    """恢复目标版本不存在（恢复被拒绝，生效版本不变）。"""


def restore_version(conn: sqlite3.Connection, target: int) -> int:
    """把历史版本 ``target`` 整体恢复为新的生效版本。

    在单个写事务中完成：校验目标版本存在 → 分配递增的新版本号 → 完整复制
    目标版本的节点/路段/封闭记录/反向通行窗 → 切换 ``current_version`` 指针。
    目标版本本身原样保留（历史仍可追溯），指针只前进不回拨。

    目标不存在时抛出 :class:`VersionNotFoundError` 并整体回滚，生效版本不变；
    目标版本号非法（非正整数）时抛出 :class:`ValidationError`。
    """
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValidationError(f"恢复目标版本号必须是正整数，收到 {target!r}")
    with transaction(conn):
        exists = conn.execute(
            "SELECT 1 FROM versions WHERE version_id=?", (target,)
        ).fetchone()
        if exists is None:
            raise VersionNotFoundError(f"版本 v{target} 不存在，无法恢复")
        new = int(
            conn.execute("SELECT COALESCE(MAX(version_id), 0) + 1 AS v FROM versions").fetchone()["v"]
        )
        _copy_version(conn, target, new, "restore")
        conn.execute(
            "UPDATE meta SET value=? WHERE key='current_version'", (str(new),)
        )
    return new


def get_snapshot(conn: sqlite3.Connection, version_id: Optional[int] = None) -> Optional[Snapshot]:
    """读取某个版本（默认当前版本）的完整数据。无已发布版本返回 None。"""
    if version_id is None:
        version_id = _current_version(conn)
    if not version_id:
        return None
    exists = conn.execute(
        "SELECT 1 FROM versions WHERE version_id=?", (version_id,)
    ).fetchone()
    if exists is None:
        return None

    nodes = [r["node_id"] for r in conn.execute(
        "SELECT node_id FROM nodes WHERE version_id=? ORDER BY node_id", (version_id,))]
    edges = [
        EdgeRow(r["edge_id"], r["from_node"], r["to_node"], r["travel_seconds"], bool(r["has_stairs"]))
        for r in conn.execute(
            "SELECT edge_id, from_node, to_node, travel_seconds, has_stairs "
            "FROM edges WHERE version_id=?", (version_id,))
    ]
    closures = [
        ClosureRow(r["edge_id"], r["start_ts"], r["end_ts"])
        for r in conn.execute(
            "SELECT edge_id, start_ts, end_ts FROM closures WHERE version_id=? "
            "ORDER BY closure_id", (version_id,))
    ]
    reverse_windows = [
        ReverseWindowRow(r["edge_id"], r["start_ts"], r["end_ts"])
        for r in conn.execute(
            "SELECT edge_id, start_ts, end_ts FROM reverse_windows WHERE version_id=? "
            "ORDER BY edge_id, window_idx", (version_id,))
    ]
    return Snapshot(version_id, nodes, edges, closures, reverse_windows)


def current_version(conn: sqlite3.Connection) -> int:
    return _current_version(conn)


def export_json(conn: sqlite3.Connection, path: Path) -> None:
    """辅助：把当前版本导出为 JSON（CLI 排错用）。"""
    snap = get_snapshot(conn)
    if snap is None:
        raise RuntimeError("当前没有已发布的数据版本")
    path.write_text(
        json.dumps(
            {
                "version_id": snap.version_id,
                "nodes": snap.nodes,
                "edges": [e._asdict() | {"has_stairs": e.has_stairs} for e in snap.edges],
                "closures": [c._asdict() for c in snap.closures],
                "reverse_windows": [w._asdict() for w in snap.reverse_windows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
