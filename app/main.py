"""FastAPI 应用：查询、原子导入与联合预检 HTTP API。"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .db import (
    VersionNotFoundError,
    connect,
    current_version,
    get_snapshot,
    init_db,
    publish_closures,
    publish_graph,
    restore_version,
)
from .precheck import precheck as run_precheck
from .routing import (
    MeetingVisitor,
    NoMeetingCandidateError,
    NoRouteError,
    Stop,
    build_engine,
)
from .schemas import ValidationError as PayloadValidationError
from .timeutil import format_time


def _segment_to_dict(seg) -> dict:
    return {
        "edge_id": seg.edge_id,
        "from_node": seg.from_node,
        "to_node": seg.to_node,
        "has_stairs": seg.has_stairs,
        "enter_time": format_time(seg.enter_time),
        "enter_time_epoch": seg.enter_time,
        "leave_time": format_time(seg.leave_time),
        "leave_time_epoch": seg.leave_time,
        "travel_seconds": seg.travel_seconds,
        "waiting_seconds": seg.waiting_seconds,
    }


def _stop_to_dict(visit) -> dict:
    return {
        "node": visit.node,
        "arrival_time": format_time(visit.arrival_time),
        "arrival_time_epoch": visit.arrival_time,
        "start_time": format_time(visit.start_time),
        "start_time_epoch": visit.start_time,
        "end_time": format_time(visit.end_time),
        "end_time_epoch": visit.end_time,
        "stay_seconds": visit.stay_seconds,
        "wait_for_window_seconds": visit.start_time - visit.arrival_time,
    }


def _route_to_dict(route) -> dict:
    total_stay = sum(v.stay_seconds for v in route.stops)
    result = {
        "origin": route.origin,
        "destination": route.destination,
        "query_departure_time": format_time(route.departure_time),
        "query_departure_time_epoch": route.departure_time,
        "arrival_time": format_time(route.arrival_time),
        "arrival_time_epoch": route.arrival_time,
        # 等候 = 非通行、非停留时间（封闭等候 + 等窗口开放）
        "total_waiting_seconds": route.arrival_time
        - route.departure_time
        - sum(s.travel_seconds for s in route.segments)
        - total_stay,
        "avoid_stairs": route.avoid_stairs,
        "edge_sequence": [s.edge_id for s in route.segments],
        "segments": [_segment_to_dict(s) for s in route.segments],
    }
    if route.stops:
        # 仅带停靠查询才返回停靠明细，无停靠时保持原有返回结构
        result["stops"] = [_stop_to_dict(v) for v in route.stops]
        result["total_stay_seconds"] = total_stay
    return result


def _maybe_seed(conn) -> None:
    """启动时初始化：数据库为空则按环境变量导入种子数据。"""
    if current_version(conn) != 0:
        return
    graph_path = config.SEED_GRAPH_FILE
    if graph_path:
        publish_graph(conn, json.loads(Path(graph_path).read_text(encoding="utf-8")))
    elif config.SEED_SAMPLE:
        from .sample_data import SAMPLE_CLOSURES, SAMPLE_GRAPH

        publish_graph(conn, SAMPLE_GRAPH)
    if current_version(conn) == 0:
        return

    closures_path = config.SEED_CLOSURES_FILE
    if closures_path:
        publish_closures(conn, json.loads(Path(closures_path).read_text(encoding="utf-8")))
    elif not graph_path and config.SEED_SAMPLE:
        from .sample_data import SAMPLE_CLOSURES

        publish_closures(conn, SAMPLE_CLOSURES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 每个请求使用独立连接（SQLite 连接轻量，且避免跨线程共享）
    yield


app = FastAPI(
    title="园区访客通行规划服务",
    version="1.0.0",
    description="离线步道图 + 半开时间区间封闭记录的最早到达路线查询",
    lifespan=lifespan,
)


@app.exception_handler(PayloadValidationError)
async def _validation_handler(request: Request, exc: PayloadValidationError):
    return JSONResponse(status_code=400, content={"error": "invalid_payload", "detail": str(exc)})


@app.get("/health")
async def health():
    conn = connect()
    try:
        init_db(conn)
        version = current_version(conn)
        return {"status": "ok", "data_version": version, "has_data": version != 0}
    finally:
        conn.close()


@app.get("/version")
async def version():
    conn = connect()
    try:
        init_db(conn)
        snap = get_snapshot(conn)
        if snap is None:
            return {"data_version": 0, "nodes": 0, "edges": 0,
                    "closures": 0, "reverse_windows": 0}
        return {
            "data_version": snap.version_id,
            "nodes": len(snap.nodes),
            "edges": len(snap.edges),
            "closures": len(snap.closures),
            "reverse_windows": len(snap.reverse_windows),
        }
    finally:
        conn.close()


@app.post("/api/route")
async def query_route(request: Request):
    """查询最早到达路线。单次请求固定使用进入时的当前数据版本。"""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400, content={"error": "invalid_payload", "detail": "请求体必须是合法 JSON"}
        )

    conn = connect()
    try:
        init_db(conn)
        # 先固定版本指针，保证整次查询只用同一版本
        snap = get_snapshot(conn)
        if snap is None:
            return JSONResponse(
                status_code=503,
                content={"error": "no_data", "detail": "尚未导入步道图，请先调用导入接口"},
            )

        from .schemas import normalize_query

        query = normalize_query(payload, snap.node_set)
        engine = build_engine(snap)
        stops = tuple(
            Stop(s["node"], s["earliest"], s["latest"], s["stay"])
            for s in query["stops"]
        )
        try:
            route = engine.plan(
                origin=query["origin"],
                destination=query["destination"],
                departure=query["departure"],
                avoid_stairs=query["avoid_stairs"],
                stops=stops,
            )
        except NoRouteError as exc:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "no_route",
                    "detail": str(exc),
                    "data_version": snap.version_id,
                },
            )

        result = _route_to_dict(route)
        result["data_version"] = snap.version_id
        return result
    finally:
        conn.close()


def _meeting_visitor_to_dict(arrival, meeting_time: int) -> dict:
    """单名访客的汇合结果：完整单人路线 + 到达后的等待秒数。"""
    route = arrival.route
    total_stay = sum(v.stay_seconds for v in route.stops)
    result = {
        "id": arrival.visitor.id,
        "origin": route.origin,
        "destination": route.destination,
        "query_departure_time": format_time(route.departure_time),
        "query_departure_time_epoch": route.departure_time,
        "arrival_time": format_time(route.arrival_time),
        "arrival_time_epoch": route.arrival_time,
        "avoid_stairs": route.avoid_stairs,
        "edge_sequence": [s.edge_id for s in route.segments],
        "segments": [_segment_to_dict(s) for s in route.segments],
        # 到达汇合点后等待全员到齐的秒数（最晚抵达者为 0）
        "wait_after_arrival_seconds": meeting_time - route.arrival_time,
        # 与单人路线查询一致的途中等候（封闭等候 + 等窗）
        "total_waiting_seconds": route.arrival_time
        - route.departure_time
        - sum(s.travel_seconds for s in route.segments)
        - total_stay,
    }
    if route.stops:
        result["stops"] = [_stop_to_dict(v) for v in route.stops]
        result["total_stay_seconds"] = total_stay
    return result


@app.post("/api/meeting")
async def plan_meeting(request: Request):
    """为多名访客从候选节点中选公共汇合点。整次计算固定同一数据版本。"""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400, content={"error": "invalid_payload", "detail": "请求体必须是合法 JSON"}
        )

    conn = connect()
    try:
        init_db(conn)
        # 与单人路线查询一致：进入时固定版本指针，整次计算只用同一版本
        snap = get_snapshot(conn)
        if snap is None:
            return JSONResponse(
                status_code=503,
                content={"error": "no_data", "detail": "尚未导入步道图，请先调用导入接口"},
            )

        from .schemas import normalize_meeting

        query = normalize_meeting(payload, snap.node_set)
        engine = build_engine(snap)
        visitors = tuple(
            MeetingVisitor(v["id"], v["origin"], v["departure"], v["avoid_stairs"])
            for v in query["visitors"]
        )
        try:
            plan = engine.plan_meeting(
                visitors=visitors,
                candidates=tuple(query["candidates"]),
                latest_meeting_time=query["latest_meeting_time"],
            )
        except NoMeetingCandidateError as exc:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "no_meeting_candidate",
                    "detail": str(exc),
                    "data_version": snap.version_id,
                },
            )

        return {
            "meeting_node": plan.node,
            "meeting_time": format_time(plan.meeting_time),
            "meeting_time_epoch": plan.meeting_time,
            "latest_meeting_time": format_time(query["latest_meeting_time"]),
            "latest_meeting_time_epoch": query["latest_meeting_time"],
            "total_arrival_seconds": plan.total_arrival_seconds,
            "visitors": [
                _meeting_visitor_to_dict(a, plan.meeting_time) for a in plan.arrivals
            ],
            "data_version": snap.version_id,
        }
    finally:
        conn.close()


@app.post("/admin/graph")
async def import_graph(request: Request):
    """原子发布新步道图（校验失败保留旧版本）。"""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400, content={"error": "invalid_payload", "detail": "请求体必须是合法 JSON"}
        )
    conn = connect()
    try:
        init_db(conn)
        old_version = current_version(conn)
        try:
            new_version = publish_graph(conn, payload)
        except PayloadValidationError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_payload",
                    "detail": str(exc),
                    "rejected": True,
                    "active_version": old_version,
                },
            )
        return {
            "status": "published",
            "data_version": new_version,
            "previous_version": old_version,
        }
    finally:
        conn.close()


@app.post("/admin/restore")
async def restore_to_version(request: Request):
    """把指定历史版本整体恢复为新的生效版本（版本号继续递增，不回拨指针）。"""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400, content={"error": "invalid_payload", "detail": "请求体必须是合法 JSON"}
        )
    conn = connect()
    try:
        init_db(conn)
        old_version = current_version(conn)
        from .schemas import normalize_restore

        try:
            target = normalize_restore(payload)
        except PayloadValidationError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_payload",
                    "detail": str(exc),
                    "rejected": True,
                    "active_version": old_version,
                },
            )
        try:
            new_version = restore_version(conn, target)
        except VersionNotFoundError as exc:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "version_not_found",
                    "detail": str(exc),
                    "rejected": True,
                    "active_version": old_version,
                },
            )
        return {
            "status": "restored",
            "source_version": target,
            "data_version": new_version,
            "previous_version": old_version,
        }
    finally:
        conn.close()


@app.post("/admin/precheck")
async def precheck_data(request: Request):
    """联合预检候选步道图与封闭记录（只读，不改变当前数据版本）。"""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400, content={"error": "invalid_payload", "detail": "请求体必须是合法 JSON"}
        )
    conn = connect()
    try:
        init_db(conn)
        try:
            report = run_precheck(payload)
        except PayloadValidationError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_payload", "detail": str(exc)},
            )
        # 只读接口：附上当前版本号，便于调用方确认预检未改动数据
        report["data_version"] = current_version(conn)
        return report
    finally:
        conn.close()


@app.post("/admin/closures")
async def import_closures(request: Request):
    """原子整体替换封闭记录（校验失败保留旧版本）。"""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400, content={"error": "invalid_payload", "detail": "请求体必须是合法 JSON"}
        )
    conn = connect()
    try:
        init_db(conn)
        old_version = current_version(conn)
        try:
            new_version = publish_closures(conn, payload)
        except PayloadValidationError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_payload",
                    "detail": str(exc),
                    "rejected": True,
                    "active_version": old_version,
                },
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "invalid_state",
                    "detail": str(exc),
                    "active_version": old_version,
                },
            )
        return {
            "status": "published",
            "data_version": new_version,
            "previous_version": old_version,
        }
    finally:
        conn.close()
