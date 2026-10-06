"""历史版本恢复（restore）测试：整体复制、版本递增、并发顺序与拒绝场景。"""
import os
import tempfile
import threading
import unittest
from pathlib import Path

from app.db import (
    VersionNotFoundError,
    connect,
    current_version,
    get_restore_source,
    get_snapshot,
    init_db,
    publish_closures,
    publish_graph,
    restore_version,
)

GRAPH_V1 = {
    "nodes": ["A", "B", "C"],
    "edges": [
        {"id": "E1", "from": "A", "to": "B", "travel_seconds": 10, "has_stairs": False},
        {"id": "E2", "from": "B", "to": "C", "travel_seconds": 20, "has_stairs": True},
    ],
}
CLOSURES_V1 = {
    "closures": [
        {"edge_id": "E1", "start": "2026-10-05T08:00:00Z", "end": "2026-10-05T09:00:00Z"}
    ]
}
CLOSURES_V2 = {
    "closures": [
        {"edge_id": "E2", "start": "2026-10-05T10:00:00Z", "end": "2026-10-05T11:00:00Z"}
    ]
}
GRAPH_V3 = {
    "nodes": ["A", "C"],
    "edges": [
        {"id": "E9", "from": "A", "to": "C", "travel_seconds": 99, "has_stairs": False,
         "reverse_windows": [
             {"start": "2026-10-05T11:00:00Z", "end": "2026-10-05T12:00:00Z"}]}
    ],
}


class RestoreTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test.db")
        os.environ["APP_DB_PATH"] = self.db_path
        self.conn = connect(Path(self.db_path))
        init_db(self.conn)
        # v1: 图 V1；v2: 图 V1 + 封闭 V1；v3: 图 V1 + 封闭 V2；
        # v4: 图 V3（含 1 条反向窗，E1/E2 消失，封闭记录随悬挂清理为空）
        publish_graph(self.conn, GRAPH_V1)
        publish_closures(self.conn, CLOSURES_V1)
        publish_closures(self.conn, CLOSURES_V2)
        publish_graph(self.conn, GRAPH_V3)
        self.assertEqual(current_version(self.conn), 4)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()
        os.environ.pop("APP_DB_PATH", None)


def _snapshot_sig(snap) -> tuple:
    return (
        tuple(snap.nodes),
        tuple((e.id, e.src, e.dst, e.seconds, e.has_stairs) for e in snap.edges),
        tuple((c.edge_id, c.start, c.end) for c in snap.closures),
        tuple((w.edge_id, w.start, w.end) for w in snap.reverse_windows),
    )


class RestoreTests(RestoreTestBase):
    def test_restore_creates_new_incrementing_version(self):
        new, source = restore_version(self.conn, 2)
        self.assertEqual(source, 2)
        self.assertEqual(new, 5)  # 版本号继续递增，而不是倒拨到 2
        self.assertEqual(current_version(self.conn), 5)

    def test_restore_copies_all_four_kinds_from_source(self):
        new, _ = restore_version(self.conn, 2)
        cur = get_snapshot(self.conn)
        src = get_snapshot(self.conn, 2)
        self.assertEqual(cur.version_id, new)
        self.assertEqual(_snapshot_sig(cur), _snapshot_sig(src))
        # 恢复版本带独立版本号，但通行数据与来源完全一致
        self.assertEqual(cur.nodes, ["A", "B", "C"])
        self.assertEqual([c.edge_id for c in cur.closures], ["E1"])
        self.assertEqual(cur.reverse_windows, [])

    def test_restore_brings_back_graph_closures_and_reverse_windows(self):
        # v4 是 GRAPH_V3：节点只有 A/C、边 E9、无封闭、1 条反向窗
        v4 = get_snapshot(self.conn, 4)
        self.assertEqual(v4.nodes, ["A", "C"])
        self.assertEqual([e.id for e in v4.edges], ["E9"])
        self.assertEqual(v4.closures, [])
        self.assertEqual(len(v4.reverse_windows), 1)

        restore_version(self.conn, 1)  # 恢复到最初只有图、无封闭的版本
        cur = get_snapshot(self.conn)
        self.assertEqual(current_version(self.conn), 5)
        self.assertEqual(cur.nodes, ["A", "B", "C"])
        self.assertEqual([e.id for e in cur.edges], ["E1", "E2"])
        self.assertEqual(cur.closures, [])
        self.assertEqual(cur.reverse_windows, [])

        restore_version(self.conn, 4)  # 再恢复含反向窗的 v4
        cur = get_snapshot(self.conn)
        self.assertEqual(current_version(self.conn), 6)
        self.assertEqual([e.id for e in cur.edges], ["E9"])
        self.assertEqual(len(cur.reverse_windows), 1)
        self.assertEqual(
            (cur.reverse_windows[0].edge_id, cur.reverse_windows[0].start,
             cur.reverse_windows[0].end),
            (v4.reverse_windows[0].edge_id, v4.reverse_windows[0].start,
             v4.reverse_windows[0].end),
        )

    def test_history_remains_traceable_and_source_recorded(self):
        new, _ = restore_version(self.conn, 2)
        # 来源版本与全部历史版本仍可读取
        for vid in (1, 2, 3, 4, 5):
            self.assertIsNotNone(get_snapshot(self.conn, vid), vid)
        self.assertEqual(get_restore_source(self.conn, new), 2)
        # 普通发布版本不是恢复版本
        self.assertIsNone(get_restore_source(self.conn, 1))
        self.assertIsNone(get_restore_source(self.conn, 4))

    def test_restoring_current_version_is_a_full_copy(self):
        # 恢复当前版本：内容相同，仍是一个新版本（指针前进，不倒拨）
        new, source = restore_version(self.conn, 4)
        self.assertEqual((new, source), (5, 4))
        self.assertEqual(current_version(self.conn), 5)
        self.assertEqual(
            _snapshot_sig(get_snapshot(self.conn, 5)),
            _snapshot_sig(get_snapshot(self.conn, 4)),
        )
        self.assertEqual(get_restore_source(self.conn, 5), 4)

    def test_unknown_version_rejected_without_changing_active(self):
        with self.assertRaises(VersionNotFoundError):
            restore_version(self.conn, 999)
        with self.assertRaises(VersionNotFoundError):
            restore_version(self.conn, 5)  # 尚未分配的下一号
        self.assertEqual(current_version(self.conn), 4)
        # 没有留下任何半成品版本
        self.assertIsNone(get_snapshot(self.conn, 5))

    def test_invalid_argument_rejected_without_changing_active(self):
        for bad in (0, -1, "2", 2.0, True, None, [2]):
            with self.assertRaises((ValueError, TypeError)):
                restore_version(self.conn, bad)
            self.assertEqual(current_version(self.conn), 4, bad)
        self.assertIsNone(get_restore_source(self.conn, 5))

    def test_publish_after_restore_builds_on_restored_snapshot(self):
        restore_version(self.conn, 2)  # v5 == v2（图V1 + E1 封闭）
        # 恢复之后继续发布封闭记录，应以恢复版本为基线
        v6 = publish_closures(self.conn, {"closures": []})
        self.assertEqual(v6, 6)
        cur = get_snapshot(self.conn)
        self.assertEqual([e.id for e in cur.edges], ["E1", "E2"])  # 图随恢复回来
        self.assertEqual(cur.closures, [])
        # 且恢复版本 v5 仍是发布前的完整快照
        v5 = get_snapshot(self.conn, 5)
        self.assertEqual([c.edge_id for c in v5.closures], ["E1"])


class RestoreConcurrencyTests(RestoreTestBase):
    def test_concurrent_writers_serialize_into_unique_versions(self):
        path = self.db_path
        errors = []
        results = []

        def restore_later():
            try:
                c = connect(Path(path))
                try:
                    new, source = restore_version(c, 2)
                    results.append(("restored", new, source))
                finally:
                    c.close()
            except BaseException as exc:  # 记录线程内任何失败
                errors.append(exc)

        def publish_other():
            try:
                c = connect(Path(path))
                try:
                    results.append(("published", publish_closures(c, CLOSURES_V1)))
                finally:
                    c.close()
            except BaseException as exc:
                errors.append(exc)
        threads = [
            threading.Thread(target=restore_later),
            threading.Thread(target=publish_other),
            threading.Thread(target=restore_later),
            threading.Thread(target=restore_later),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
            self.assertFalse(t.is_alive(), "写操作线程超时（未串行完成）")

        self.assertEqual(errors, [])
        new_ids = sorted(v[1] for v in results)
        # 四个并发写各得一个新版本号 5/6/7/8，无重复、无空缺
        self.assertEqual(new_ids, [5, 6, 7, 8])
        self.assertEqual(current_version(self.conn), 8)
        self.assertEqual(sum(1 for v in results if v[0] == "restored"), 3)

    def test_no_torn_version_visible_to_snapshot_reads(self):
        """每个已存在版本的快照必须始终自洽（无跨版本拼接）。"""
        path = self.db_path
        stop = threading.Event()
        bad = []

        def reader():
            c = connect(Path(path))
            try:
                while not stop.is_set():
                    for vid in range(1, current_version(c) + 1):
                        snap = get_snapshot(c, vid)
                        if snap is None:
                            continue
                        known = {e.id for e in snap.edges}
                        # 封闭/反向窗只可引用本版本存在的边
                        for cl in snap.closures:
                            if cl.edge_id not in known:
                                bad.append((vid, "closure", cl.edge_id))
                        for w in snap.reverse_windows:
                            if w.edge_id not in known:
                                bad.append((vid, "window", w.edge_id))
            finally:
                c.close()

        writers = []

        def do_restore():
            c = connect(Path(path))
            try:
                restore_version(c, 2)
            finally:
                c.close()

        def do_graph():
            c = connect(Path(path))
            try:
                publish_graph(c, GRAPH_V1)
            finally:
                c.close()

        r = threading.Thread(target=reader)
        r.start()
        for _ in range(3):
            writers.append(threading.Thread(target=do_restore))
            writers.append(threading.Thread(target=do_graph))
        for t in writers:
            t.start()
        for t in writers:
            t.join(10)
        stop.set()
        r.join(10)
        self.assertEqual(bad, [])
        # 最终再做一次全版本自洽检查
        c = connect(Path(path))
        try:
            for vid in range(1, current_version(c) + 1):
                snap = get_snapshot(c, vid)
                known = {e.id for e in snap.edges}
                self.assertTrue({cl.edge_id for cl in snap.closures} <= known, vid)
                self.assertTrue({w.edge_id for w in snap.reverse_windows} <= known, vid)
        finally:
            c.close()


class RestoreMigrationTests(unittest.TestCase):
    def test_old_schema_db_upgrades_and_restore_works(self):
        """早期数据库（versions.kind 无 'restore'、无 restore_events 表）可在线升级。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "old.db"
        conn = connect(path)
        # 手工建一份“旧库”：旧的 kind CHECK，无 restore_events 表
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta VALUES ('current_version', '0');
            CREATE TABLE versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL CHECK (kind IN ('graph','closures')),
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')));
            CREATE TABLE nodes (version_id INTEGER NOT NULL, node_id TEXT NOT NULL,
                PRIMARY KEY (version_id, node_id));
            CREATE TABLE edges (version_id INTEGER NOT NULL, edge_id TEXT NOT NULL,
                from_node TEXT NOT NULL, to_node TEXT NOT NULL,
                travel_seconds INTEGER NOT NULL CHECK (travel_seconds > 0),
                has_stairs INTEGER NOT NULL CHECK (has_stairs IN (0,1)),
                PRIMARY KEY (version_id, edge_id));
            CREATE TABLE closures (version_id INTEGER NOT NULL, closure_id INTEGER NOT NULL,
                edge_id TEXT NOT NULL, start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL
                CHECK (end_ts > start_ts), PRIMARY KEY (version_id, closure_id));
            CREATE TABLE reverse_windows (version_id INTEGER NOT NULL, edge_id TEXT NOT NULL,
                window_idx INTEGER NOT NULL, start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL
                CHECK (end_ts > start_ts), PRIMARY KEY (version_id, edge_id, window_idx));
            INSERT INTO versions(version_id, kind) VALUES (1, 'graph');
            INSERT INTO nodes VALUES (1, 'A'), (1, 'B');
            INSERT INTO edges VALUES (1, 'E1', 'A', 'B', 10, 0);
            UPDATE meta SET value='1' WHERE key='current_version';
            """
        )
        conn.close()

        os.environ["APP_DB_PATH"] = str(path)
        self.addCleanup(os.environ.pop, "APP_DB_PATH", None)

        conn = connect(path)
        init_db(conn)  # 触发升级
        try:
            self.assertEqual(current_version(conn), 1)
            # 历史版本号原样保留，恢复版本号从其后继续递增
            new, source = restore_version(conn, 1)
            self.assertEqual((new, source), (2, 1))
            self.assertEqual(get_restore_source(conn, 2), 1)
            snap = get_snapshot(conn)
            self.assertEqual(snap.nodes, ["A", "B"])
            self.assertEqual([e.id for e in snap.edges], ["E1"])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
