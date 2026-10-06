"""通行规划引擎。

时间规则
========
封闭记录是半开区间 ``[start, end)``，路段通行区间也是半开
``[enter, leave)``（``leave = enter + travel_seconds``）。
二者重叠当且仅当::

    enter < close_end and close_start < leave

因此允许恰好在封闭结束时刻 ``end`` 进入路段，也允许在 ``end`` 之前
停在节点上等候；通行区间一旦开始就不能与任何封闭区间重叠。

反向通行时间窗
==============
每条路段可携带一组互不重叠的半开时间窗。对从 ``from`` 到 ``to`` 的
有向路段：

- **进入时刻**落在某个窗 ``[start, end)`` 内时，只允许**反向**
  （从原终点 ``to`` 走向原起点 ``from``）；
- 其余时刻只允许原方向；
- 方向只在**进入**路段的时刻判定；进入后若通行途中跨过窗边界
  （包括开始与结束边界），仍按进入时确定的方向走完全程；
- 封闭记录对两个方向同样生效：整个通行区间 ``[enter, leave)`` 仍
  不得与该边任何封闭区间重叠，可在节点等待到可行时刻再进入；
- 未配置反向窗的路段行为不变。

上述判定给出一个关于就绪时刻单调非降（FIFO）的最早进入函数，故
时间依赖 Dijkstra 与多标签选路的正确性不变。

算法
====
第一阶段：时间依赖的标号设定（Dijkstra 变种）。
``earliest_enter`` 在 FIFO 假设下（晚出发不会更早通过同一条边）
保证最早到达时刻正确，得到终点最早到达 ``T``。

第二阶段：在“到达终点不晚于 T”的状态空间做多标号搜索，状态键为
``(到达时刻, 路段编号序列)``，按该二元组的字典序出堆。边的通行秒数
为正，每次扩展到达时刻严格增大，所以终点第一次出堆即为
“最早到达、同到达时刻路段编号序列字典序最小”的路线。

为什么不能只做一次 Dijkstra
---------------------------
节点等待会把较晚到达的路径“拉平”到相同的后续出发时刻，于是在某个
中间节点上“时刻更优”的路径，续接同一条后缀后可能与另一条路径同时
到达终点，而路段编号序列反而更大。故字典序平局判定必须在第二阶段
对完整序列进行。

按顺序停靠
==========
查询可携带一组**按顺序**完成的停靠点，每站给出开始停留的闭区间
``[earliest, latest]`` 与非负整数停留秒数：

- 抵达某站后可在节点等到窗口开放，开始停留不得晚于 ``latest``，
  结束停留后才能前往下一站；
- 途经尚未轮到的站点（包括未来各站的节点）不算完成停靠，该节点
  此时只当作普通节点通过；
- 起点本身就是第 1 站、或相邻两站位于同一节点时，立即（链式）完成停靠。

选路目标不变：**最终到达**（完成全部停靠并抵达终点）最早优先，平局
按**全程**路段编号序列字典序选取——不得逐段独立打破平局。停靠窗口
的“拉平”效应（晚到但在窗口下界前到达，结束停留时刻相同）意味着
逐段最优拼接不等于全程最优，因此第二阶段在 ``(节点, 已完成停靠数)``
状态空间上做多标签搜索，标签仍为 ``(时刻, 全程编号序列)``。
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Optional

from .db import Snapshot


@dataclass(frozen=True)
class Edge:
    id: str
    src: str
    dst: str
    seconds: int
    has_stairs: bool


class Segment(NamedTuple):
    edge_id: str
    from_node: str
    to_node: str
    enter_time: int          # 进入该路段的时刻（在起点等候后的实际出发时刻）
    leave_time: int          # 离开该路段（到达下一节点）的时刻
    travel_seconds: int
    waiting_seconds: int     # 进入前在 from_node 的等候秒数
    has_stairs: bool


class Route(NamedTuple):
    origin: str
    destination: str
    departure_time: int      # 用户请求的出发时刻
    arrival_time: int        # 最早到达时刻（完成全部停靠并抵达终点）
    avoid_stairs: bool
    segments: tuple[Segment, ...]
    stops: tuple["StopVisit", ...] = ()   # 各停靠点的抵达/开始/结束记录


class Stop(NamedTuple):
    """一个按顺序停靠点（查询输入，时刻为纪元秒）。"""

    node: str
    earliest: int   # 最早开始停留时刻（闭区间下界）
    latest: int     # 最晚开始停留时刻（闭区间上界）
    stay: int       # 停留秒数（非负整数）


class StopVisit(NamedTuple):
    """一个停靠点的实际停靠记录。"""

    node: str
    arrival_time: int   # 抵达该站时刻
    start_time: int     # 实际开始停留时刻（max(抵达, 窗口下界)）
    end_time: int       # 结束停留时刻（start + stay_seconds）
    stay_seconds: int


class NoRouteError(Exception):
    """没有在约束下可行的路线。"""


class MeetingVisitor(NamedTuple):
    """一名汇合查询访客（查询输入，时刻为纪元秒）。"""

    id: object            # 访客标识（未给 id 时为 1 起始序号）
    origin: str
    departure: int        # 最早出发时刻
    avoid_stairs: bool


class VisitorArrival(NamedTuple):
    """一名访客到某候选点的最早到达结果。"""

    visitor: MeetingVisitor
    route: Route          # 复用既有单人路线查询语义（含全程字典序破平局）


class MeetingPlan(NamedTuple):
    """一次多人汇合点规划的选中结果。"""

    node: str                              # 选中的汇合节点
    meeting_time: int                      # 全员实际汇合时刻（最晚个人抵达）
    arrivals: tuple[VisitorArrival, ...]   # 与请求 visitors 同序
    total_arrival_seconds: int             # 个人抵达时刻总和（二级择优指标）


class NoMeetingCandidateError(Exception):
    """没有任何候选点能让所有人在最晚时刻前抵达。"""


def _overlaps(enter: int, leave: int, start: int, end: int) -> bool:
    """[enter, leave) 与 [start, end) 是否重叠。"""
    return enter < end and start < leave


class RouteEngine:
    def __init__(self, snapshot: Snapshot):
        self.version_id = snapshot.version_id
        self._adj: dict[str, list[Edge]] = defaultdict(list)
        self._rev_adj: dict[str, list[Edge]] = defaultdict(list)
        self._edges: dict[str, Edge] = {}
        for row in snapshot.edges:
            edge = Edge(row.id, row.src, row.dst, row.seconds, row.has_stairs)
            if edge.id in self._edges:
                raise ValueError(f"路段编号在版本内重复: {edge.id}")
            self._edges[edge.id] = edge
            self._adj[edge.src].append(edge)
            if edge.dst != edge.src:
                # 自环的反向即原方向本身，不重复登记
                self._rev_adj[edge.dst].append(edge)
        for edges in self._adj.values():
            edges.sort(key=lambda e: e.id)
        for edges in self._rev_adj.values():
            edges.sort(key=lambda e: e.id)

        self._closures: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for c in snapshot.closures:
            self._closures[c.edge_id].append((c.start, c.end))
        for intervals in self._closures.values():
            intervals.sort()

        self._reverse_windows: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for w in snapshot.reverse_windows:
            self._reverse_windows[w.edge_id].append((w.start, w.end))
        for intervals in self._reverse_windows.values():
            intervals.sort()

    def _options(self, node: str, avoid_stairs: bool) -> list[tuple[Edge, bool]]:
        """从 ``node`` 可扩展的 ``(路段, 是否反向)`` 列表。

        反向选项是否真正可用由 :meth:`earliest_enter` 按进入时刻判定；
        这里只给出“在某段时间窗内可能可用”的边（即配置了反向窗的入边）。
        """
        options: list[tuple[Edge, bool]] = [
            (e, False) for e in self._adj.get(node, ())
            if not (avoid_stairs and e.has_stairs)
        ]
        for e in self._rev_adj.get(node, ()):
            if avoid_stairs and e.has_stairs:
                continue
            if not self._reverse_windows.get(e.id):
                continue  # 未配置反向窗：反向恒不可用
            options.append((e, True))
        options.sort(key=lambda item: (item[0].id, item[1]))
        return options

    def earliest_departure(self, edge: Edge, ready_at: int) -> int:
        """原方向在 ``ready_at`` 或之后最早可以进入该路段的时刻（兼容接口）。

        若按当前时刻进入会与某条封闭区间重叠，则把进入时刻推迟到该封闭
        区间结束（半开区间，结束时刻立即可通行）后重新检查。
        """
        enter = self.earliest_enter(edge, ready_at, reverse=False)
        if enter is None:  # 无反向窗时不会发生；防御性处理
            raise NoRouteError("内部错误：原方向无可用时间窗")
        return enter

    def earliest_enter(self, edge: Edge, ready_at: int, reverse: bool) -> Optional[int]:
        """在 ``ready_at`` 或之后，按指定方向最早可以进入该路段的时刻。

        同时满足两类约束：

        1. 封闭记录：整个通行区间 ``[enter, enter + seconds)`` 不得与该边
           任何封闭区间重叠；可在节点等待封闭结束；
        2. 反向时间窗：反向通行的进入时刻必须落在某个
           ``[win_start, win_end)`` 内，原方向的进入时刻必须落在所有
           时间窗之外（等待到窗结束）。未配置反向窗时两个方向都只有
           原方向可行。

        返回最早进入时刻；该方向此后无可用时间窗时返回 ``None``。
        方向只按进入时刻判定，故等待封闭而越过窗边界时需要重新选向。
        """
        t = ready_at
        windows = self._reverse_windows.get(edge.id, ())
        while True:
            # 1) 封闭约束：等待到通行区间完全不与任何封闭区间重叠
            blocked = False
            for start, end in self._closures.get(edge.id, ()):
                if end <= t:
                    continue
                if start >= t + edge.seconds:
                    break  # 已按 start 排序，后面的封闭更不可能重叠
                if _overlaps(t, t + edge.seconds, start, end):
                    t = end
                    blocked = True
                    break
            if blocked:
                continue  # 越过封闭结束时刻后，可能也越过了窗边界，重检
            # 2) 反向时间窗约束（窗对两个方向都强制选向）
            if not windows:
                return None if reverse else t
            containing = None  # t 所在的窗（半开）
            next_start = None  # t 之后第一个开始的窗
            for win_start, win_end in windows:
                if win_end <= t:
                    continue
                if win_start <= t < win_end:
                    containing = (win_start, win_end)
                elif win_start > t:
                    next_start = win_start
                    break  # 已按 start 排序
            if reverse:
                if containing is not None:
                    return t
                if next_start is None:
                    return None  # 之后再无反向窗
                t = next_start  # 在节点等待下一反向窗开放
            else:
                if containing is None:
                    return t
                t = containing[1]  # 窗内原方向禁行，等待窗结束
            # 跨越窗边界后重新检查封闭区间

    # ------------------------------------------------------------------
    # 第一阶段：最早到达时刻
    # ------------------------------------------------------------------
    def _earliest_arrival_map(
        self, origin: str, departure: int, avoid_stairs: bool
    ) -> dict[str, int]:
        dist: dict[str, int] = {origin: departure}
        heap: list[tuple[int, str]] = [(departure, origin)]
        while heap:
            t, u = heapq.heappop(heap)
            if t != dist[u]:
                continue
            for edge, reversed_dir in self._options(u, avoid_stairs):
                enter = self.earliest_enter(edge, t, reversed_dir)
                if enter is None:
                    continue
                v = edge.src if reversed_dir else edge.dst
                arrival = enter + edge.seconds
                if arrival < dist.get(v, float("inf")):
                    dist[v] = arrival
                    heapq.heappush(heap, (arrival, v))
        return dist

    # ------------------------------------------------------------------
    # 第二阶段：最早到达前提下，路段编号序列字典序最小
    # ------------------------------------------------------------------
    @staticmethod
    def _dominates(t1: int, s1: tuple[str, ...], t2: int, s2: tuple[str, ...]) -> bool:
        """标签 1 是否支配标签 2（对任意同后缀都不会更差）。

        两条规则，都要避开“前缀陷阱”——较短序列追加后缀后字典序可能翻转：

        1. 到达时刻相同：正秒数下图中不存在零耗时环，同一节点上同到达
           时刻的两条不同序列不可能互为前缀，可直接按字典序裁剪；
        2. 到达时刻严格更早：只有**等长**序列才安全（等长时首个不同
           元素的比较与追加同后缀后的完整比较一致）；长度不同且互为
           前缀的必须同时保留。
        """
        if t1 > t2:
            return False
        if t1 == t2:
            return s1 < s2
        # t1 < t2
        if len(s1) != len(s2):
            return False
        return s1 < s2

    def _lex_smallest(
        self,
        origin: str,
        destination: str,
        departure: int,
        avoid_stairs: bool,
        deadline: int,
    ) -> tuple[str, ...]:
        """在所有“到达终点不晚于 deadline”的路线中取序列字典序最小者。

        不能在终点标签第一次出堆时返回：堆键 ``(到达时刻, 序列)`` 只在
        同一节点间有可比性，而时刻更小的中间节点标签可能继续扩展出更优的
        终点标签。故完整展开搜索（秒数为正 + deadline 剪枝保证终止），
        最后在终点的标签集合里取二元组最小项。
        """
        # 每个节点保留一组互不支配的标签
        labels: dict[str, list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
        labels[origin].append((departure, ()))
        heap: list[tuple[int, tuple[str, ...], str]] = [(departure, (), origin)]

        while heap:
            t, seq, u = heapq.heappop(heap)
            if (t, seq) not in labels.get(u, ()):
                continue  # 已被支配后移除的过期标签

            for edge, reversed_dir in self._options(u, avoid_stairs):
                enter = self.earliest_enter(edge, t, reversed_dir)
                if enter is None:
                    continue
                v = edge.src if reversed_dir else edge.dst
                leave = enter + edge.seconds
                if leave > deadline:
                    continue  # 不可能参与“最早到达 T”的路线
                new_seq = seq + (edge.id,)
                new_label = (leave, new_seq)

                existing = labels[v]
                if any(self._dominates(et, es, leave, new_seq) for et, es in existing):
                    continue
                # 新标签支配旧标签则移除（前缀不可比的保留）
                labels[v] = [
                    (et, es) for et, es in existing
                    if not self._dominates(leave, new_seq, et, es)
                ]
                labels[v].append(new_label)
                heapq.heappush(heap, (leave, new_seq, v))

        best = min(labels.get(destination, ()), default=None, key=lambda item: (item[0], item[1]))
        if best is None:
            raise NoRouteError("内部错误：最早到达已知，但字典序选路失败")
        return best[1]

    # ------------------------------------------------------------------
    def plan(
        self,
        origin: str,
        destination: str,
        departure: int,
        avoid_stairs: bool = False,
        stops: tuple[Stop, ...] = (),
    ) -> Route:
        if stops:
            return self._plan_with_stops(origin, destination, departure, avoid_stairs, stops)

        dist = self._earliest_arrival_map(origin, departure, avoid_stairs)
        if destination not in dist:
            raise NoRouteError(
                "在当前步道图与封闭记录下不存在可行路线"
                + ("（已避开台阶路段）" if avoid_stairs else "")
            )
        deadline = dist[destination]
        best_seq = self._lex_smallest(
            origin, destination, departure, avoid_stairs, deadline
        )

        segments: list[Segment] = []
        current = origin
        clock = departure
        for edge_id in best_seq:
            edge = self._edges[edge_id]
            # 方向按当前所在节点判定（重放进入时刻与搜索阶段一致）
            reversed_dir = current == edge.dst and current != edge.src
            enter = self.earliest_enter(edge, clock, reversed_dir)
            if enter is None:
                raise NoRouteError("内部错误：获胜序列在重放时不可行")
            next_node = edge.src if reversed_dir else edge.dst
            leave = enter + edge.seconds
            segments.append(
                Segment(
                    edge_id=edge.id,
                    from_node=current,
                    to_node=next_node,
                    enter_time=enter,
                    leave_time=leave,
                    travel_seconds=edge.seconds,
                    waiting_seconds=enter - clock,
                    has_stairs=edge.has_stairs,
                )
            )
            clock = leave
            current = next_node

        return Route(
            origin=origin,
            destination=destination,
            departure_time=departure,
            arrival_time=deadline,
            avoid_stairs=avoid_stairs,
            segments=tuple(segments),
        )

    # ------------------------------------------------------------------
    # 按顺序停靠：窗口应用与两阶段选路
    # ------------------------------------------------------------------
    @staticmethod
    def _stays_at(node: str, k: int, clock: int, stops: tuple[Stop, ...]):
        """在 ``node`` 上链式完成所有“当前轮到且地点即 node”的停靠。

        返回 ``(k', clock', visits)``；若某站无法按窗口停留则返回
        ``(k, clock, None)`` 表示该走法不可行。
        """
        visits: list[StopVisit] = []
        while k < len(stops) and stops[k].node == node:
            stop = stops[k]
            start = max(clock, stop.earliest)
            if start > stop.latest:
                return k, clock, None
            end = start + stop.stay
            visits.append(StopVisit(node, clock, start, end, stop.stay))
            clock = end
            k += 1
        return k, clock, visits

    def _earliest_final_with_stops(
        self,
        origin: str,
        destination: str,
        departure: int,
        avoid_stairs: bool,
        stops: tuple[Stop, ...],
    ) -> int:
        """第一阶段：完成全部停靠并抵达终点的最早时刻。

        各腿的最早到达映射与“停留结束时刻”都是关于出发时刻的单调
        非降函数，其复合亦然，故逐腿取最早到达即得全局最早最终到达；
        同时最早到达时若窗口已关，则更晚到达也不可能满足窗口。
        """
        suffix = "（已避开台阶路段）" if avoid_stairs else ""
        clock = departure
        current = origin
        k = 0
        while True:
            # 当前节点即下一停靠点时立即（链式）完成停靠
            k, clock, visits = self._stays_at(current, k, clock, stops)
            if visits is None:
                raise NoRouteError(
                    f"第 {k + 1} 站（{stops[k].node}）无法按时停留："
                    f"开始停留不得晚于该站最晚时刻"
                )
            if k >= len(stops):
                break
            stop = stops[k]
            dist = self._earliest_arrival_map(current, clock, avoid_stairs)
            if stop.node not in dist:
                raise NoRouteError(f"无法到达第 {k + 1} 站（{stop.node}）{suffix}")
            clock = dist[stop.node]
            current = stop.node
        if current != destination:
            dist = self._earliest_arrival_map(current, clock, avoid_stairs)
            if destination not in dist:
                raise NoRouteError(
                    "在当前步道图与封闭记录下不存在可行路线" + suffix
                )
            clock = dist[destination]
        return clock

    def _lex_smallest_with_stops(
        self,
        origin: str,
        destination: str,
        departure: int,
        avoid_stairs: bool,
        stops: tuple[Stop, ...],
        deadline: int,
    ) -> tuple[str, ...]:
        """在“最终到达不晚于 deadline”的走法中取全程编号序列字典序最小者。

        状态为 ``(节点, 已完成停靠数)``；抵达当前轮到的站点即强制完成
        停靠（可链式），途经未来站点不产生停靠。支配规则与单段情形相同，
        因为同一状态下任意共同后缀的窗口/封闭行为一致。
        """
        k0, t0, visits0 = self._stays_at(origin, 0, departure, stops)
        if visits0 is None:
            raise NoRouteError("第 1 站即起点，但出发时刻已错过其最晚开始停留时刻")

        labels: dict[tuple[str, int], list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
        labels[(origin, k0)].append((t0, ()))
        heap: list[tuple[int, tuple[str, ...], str, int]] = [(t0, (), origin, k0)]

        while heap:
            t, seq, u, k = heapq.heappop(heap)
            if (t, seq) not in labels.get((u, k), ()):
                continue  # 已被支配后移除的过期标签

            for edge, reversed_dir in self._options(u, avoid_stairs):
                enter = self.earliest_enter(edge, t, reversed_dir)
                if enter is None:
                    continue
                v = edge.src if reversed_dir else edge.dst
                leave = enter + edge.seconds
                if leave > deadline:
                    continue  # 不可能参与“最早到达”的路线
                nk, nt = k, leave
                if nk < len(stops) and stops[nk].node == v:
                    nk, nt, visits = self._stays_at(v, nk, leave, stops)
                    if visits is None or nt > deadline:
                        continue  # 窗口不可行或注定超过最早到达
                new_seq = seq + (edge.id,)
                new_label = (nt, new_seq)

                state = (v, nk)
                existing = labels[state]
                if any(self._dominates(et, es, nt, new_seq) for et, es in existing):
                    continue
                labels[state] = [
                    (et, es) for et, es in existing
                    if not self._dominates(nt, new_seq, et, es)
                ]
                labels[state].append(new_label)
                heapq.heappush(heap, (nt, new_seq, v, nk))

        best = min(
            labels.get((destination, len(stops)), ()),
            default=None,
            key=lambda item: (item[0], item[1]),
        )
        if best is None:
            raise NoRouteError("内部错误：最早到达已知，但字典序选路失败")
        return best[1]

    def _plan_with_stops(
        self,
        origin: str,
        destination: str,
        departure: int,
        avoid_stairs: bool,
        stops: tuple[Stop, ...],
    ) -> Route:
        deadline = self._earliest_final_with_stops(
            origin, destination, departure, avoid_stairs, stops
        )
        best_seq = self._lex_smallest_with_stops(
            origin, destination, departure, avoid_stairs, stops, deadline
        )

        # 按获胜序列重放，生成逐段时间与各站停靠记录
        segments: list[Segment] = []
        visits: list[StopVisit] = []
        current = origin
        clock = departure
        k = 0
        k, clock, v0 = self._stays_at(origin, k, clock, stops)
        visits.extend(v0)
        for edge_id in best_seq:
            edge = self._edges[edge_id]
            reversed_dir = current == edge.dst and current != edge.src
            enter = self.earliest_enter(edge, clock, reversed_dir)
            if enter is None:
                raise NoRouteError("内部错误：获胜序列在重放时不可行")
            next_node = edge.src if reversed_dir else edge.dst
            leave = enter + edge.seconds
            segments.append(
                Segment(
                    edge_id=edge.id,
                    from_node=current,
                    to_node=next_node,
                    enter_time=enter,
                    leave_time=leave,
                    travel_seconds=edge.seconds,
                    waiting_seconds=enter - clock,
                    has_stairs=edge.has_stairs,
                )
            )
            clock = leave
            current = next_node
            if k < len(stops) and stops[k].node == current:
                k, clock, vk = self._stays_at(current, k, clock, stops)
                visits.extend(vk)

        return Route(
            origin=origin,
            destination=destination,
            departure_time=departure,
            arrival_time=deadline,
            avoid_stairs=avoid_stairs,
            segments=tuple(segments),
            stops=tuple(visits),
        )

    # ------------------------------------------------------------------
    # 多人汇合点：候选点筛选与择优
    # ------------------------------------------------------------------
    def plan_meeting(
        self,
        visitors: tuple[MeetingVisitor, ...],
        candidates: tuple[str, ...],
        latest_meeting_time: int,
    ) -> MeetingPlan:
        """在候选节点中选出全员可在最晚时刻前抵达的汇合点。

        每名访客从各自最早出发时刻独立通行（封闭等候、避台阶、最早到达
        及全程路段编号序列字典序破平局的单人路线语义不变），抵达后可在
        汇合点等待。只接受**所有**访客抵达时刻均不晚于
        ``latest_meeting_time`` 的候选点。

        合格候选按以下顺序择优：
        1. 全员实际汇合时刻（最晚个人抵达时刻）最早；
        2. 个人抵达时刻总和最小；
        3. 候选节点编号字典序最小。
        """
        # (汇合时刻, 抵达总和, 节点编号) 三级字典序即择优顺序
        best_key: Optional[tuple[int, int, str]] = None
        best_arrivals: Optional[tuple[VisitorArrival, ...]] = None
        best_node: Optional[str] = None

        for node in candidates:
            arrivals: list[VisitorArrival] = []
            feasible = True
            for visitor in visitors:
                try:
                    route = self.plan(
                        origin=visitor.origin,
                        destination=node,
                        departure=visitor.departure,
                        avoid_stairs=visitor.avoid_stairs,
                    )
                except NoRouteError:
                    feasible = False
                    break
                if route.arrival_time > latest_meeting_time:
                    feasible = False
                    break
                arrivals.append(VisitorArrival(visitor, route))
            if not feasible:
                continue

            meeting_time = max(a.route.arrival_time for a in arrivals)
            total = sum(a.route.arrival_time for a in arrivals)
            key = (meeting_time, total, node)
            if best_key is None or key < best_key:
                best_key = key
                best_node = node
                best_arrivals = tuple(arrivals)

        if best_key is None:
            raise NoMeetingCandidateError(
                "没有任何候选节点能让所有访客在最晚汇合时刻前抵达"
            )
        return MeetingPlan(
            node=best_node,
            meeting_time=best_key[0],
            arrivals=best_arrivals,
            total_arrival_seconds=best_key[1],
        )


def build_engine(snapshot: Snapshot) -> RouteEngine:
    return RouteEngine(snapshot)