# src/agent/multimedia/history_api.py
"""历史会话恢复接口。

LangGraph Platform 的 `GET /threads/{id}/state` 走 `graph.aget_state()`，会重建图并
校验通道 schema。对于用**旧版图**跑出来的历史 checkpoint，这条路径会在路由函数体
之外抛 500（已实测无法拦截），导致前端"点击历史会话没反应"。

但 checkpoint 数据本身是完好的。本模块因此绕开图重建，直接用
`AsyncSqliteSaver.aget_tuple()` 读取原始 `channel_values`，向前端提供：

- ``GET /history/threads``            分页列出历史会话摘要（不含 scenes 全量）
- ``GET /history/threads/{thread_id}`` 返回单个会话的完整中间产物 + 进度 + 中断信息

设计要点：

* **只读**：以 ``file:...?mode=ro`` URI 连接，杜绝本接口意外写库，也避免与
  LangGraph Server / 本地脚本争抢写锁（WAL 下读不阻塞写）。
* **不全表反序列化**：列表接口先用一条聚合 SQL 走主键索引取出每个 thread 的最新
  ``checkpoint_id``，再只对当前页做 msgpack 反序列化，复杂度 O(limit) 而非 O(全表)。
* **序列化兜底**：返回前用 :func:`_json_safe` 递归清洗，把 datetime / bytes / 未知
  对象降级为字符串。平台路径翻车的可能原因正是序列化，这里必须主动防御。
* **错误隔离**：单个 thread 反序列化失败只标记为 ``corrupted``，不拖垮整个列表。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, Optional

import aiosqlite
import os
import re
from fastapi import APIRouter, HTTPException, Query
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .checkpointer import _resolve_db_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])

#: 列表接口中任务标题的最大长度
_TITLE_MAX = 60

#: 递归清洗的最大深度，防御异常嵌套导致栈溢出
_MAX_DEPTH = 12

#: 合并产物视图时，单线程最多回读的历史 checkpoint 版本数。
#: 原实现用 ``alist(limit=100000)`` 全量遍历，对 1GB 库存在性能与内存风险。
#: 生成产物的最新版本集中在最近几次 revision，回读最近 N 个版本已足够还原，
#: 故收敛为有限上限（避免过度限制导致骨架选不到最全版本，取一个宽松值）。
_MAX_MERGE_VERSIONS = 1000


# --------------------------------------------------------------------------- #
# 序列化清洗
# --------------------------------------------------------------------------- #
def _json_safe(value: Any, _depth: int = 0) -> Any:
    """把任意对象递归转换为 JSON 可序列化结构。

    历史 checkpoint 里可能混入 ``datetime`` / ``bytes`` / pydantic 模型 /
    dataclass 等无法直接进 JSON 的类型。与其让 FastAPI 在响应序列化阶段抛 500
    （那正是平台路径的失败方式），不如在这里主动降级。

    Args:
        value: 待清洗的任意值。
        _depth: 当前递归深度，内部使用。

    Returns:
        由 dict / list / str / int / float / bool / None 组成的安全结构。
    """
    if _depth > _MAX_DEPTH:
        return str(value)

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        # NaN / Infinity 不是合法 JSON
        return value if value == value and value not in (float("inf"), float("-inf")) else None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return _json_safe(value.value, _depth + 1)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes:{len(value)}>"

    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, _depth + 1) for v in value]

    # langgraph.types.Interrupt 等带 .value 的包装对象
    if hasattr(value, "value") and not isinstance(value, type):
        try:
            return _json_safe(value.value, _depth + 1)
        except Exception:  # noqa: BLE001 - 兜底路径不允许再抛
            pass

    if is_dataclass(value) and not isinstance(value, type):
        try:
            from dataclasses import asdict

            return _json_safe(asdict(value), _depth + 1)
        except Exception:  # noqa: BLE001
            pass

    # pydantic v2 / v1
    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _json_safe(fn(), _depth + 1)
            except Exception:  # noqa: BLE001
                pass

    return str(value)


# --------------------------------------------------------------------------- #
# 连接管理
# --------------------------------------------------------------------------- #
def _read_only_uri() -> str:
    """构造只读 sqlite URI。"""
    path = _resolve_db_path().as_posix()
    return f"file:{path}?mode=ro"


def _sync_ro_connect() -> sqlite3.Connection:
    """同步只读连接，仅用于轻量聚合查询（走索引，不反序列化）。"""
    return sqlite3.connect(_read_only_uri(), uri=True, timeout=30)


class _SaverSession:
    """`AsyncSqliteSaver` 的一次性只读会话。

    ``AsyncSqliteSaver`` 持有的 aiosqlite 连接绑定到创建它的事件循环，
    跨请求复用在 uvicorn 下容易踩到循环归属问题。历史恢复是低频操作，
    每次请求新建连接的开销可以接受，换取稳定性。
    """

    def __init__(self) -> None:
        self._conn: Optional[aiosqlite.Connection] = None

    async def __aenter__(self) -> AsyncSqliteSaver:
        self._conn = await aiosqlite.connect(_read_only_uri(), uri=True, timeout=30)
        return AsyncSqliteSaver(self._conn)

    async def __aexit__(self, *exc: Any) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:  # noqa: BLE001 - 关闭失败不应影响响应
                logger.debug("关闭历史会话只读连接失败", exc_info=True)
            self._conn = None


# --------------------------------------------------------------------------- #
# checkpoint 解析
# --------------------------------------------------------------------------- #
def _extract_interrupt(pending_writes: Optional[Iterable[Any]]) -> Optional[dict]:
    """从 ``pending_writes`` 中提取 ``__interrupt__`` payload。

    LangGraph 把 ``interrupt()`` 的 payload 以 ``__interrupt__`` 通道写入
    pending_writes，形如 ``(task_id, "__interrupt__", [Interrupt(value={...})])``。

    Args:
        pending_writes: checkpoint tuple 的 pending_writes。

    Returns:
        含 ``stage`` 字段的 payload dict；没有待处理中断时返回 ``None``。
    """
    for write in pending_writes or []:
        # write 结构为 (task_id, channel, value)
        if len(write) < 3 or write[1] != "__interrupt__":
            continue

        raw = write[2]
        candidates = raw if isinstance(raw, (list, tuple)) else [raw]
        for item in candidates:
            payload = _json_safe(item)
            if isinstance(payload, dict) and "stage" in payload:
                return payload
    return None


def _extract_next_nodes(channel_values: dict) -> list[str]:
    """从 ``branch:to:<node>`` 通道推导下一步待执行节点。

    绕开了 ``graph.aget_state()`` 就拿不到框架计算的 ``next`` 元组。但 LangGraph
    以 ``branch:to:<node>`` 通道标记待调度节点，直接解析即可还原，对展示进度足够。

    Args:
        channel_values: checkpoint 的 channel_values。

    Returns:
        下一步节点名列表，可能为空（表示已跑完或无待调度节点）。
    """
    prefix = "branch:to:"
    return sorted(
        key[len(prefix) :] for key in channel_values if key.startswith(prefix)
    )


def _derive_status(
    channel_values: dict,
    has_interrupt: bool,
    next_nodes: list[str],
    all_scenes_done: bool = False,
) -> str:
    """归纳会话状态徽标。

    Returns:
        ``interrupted`` | ``aborted`` | ``done`` | ``running`` | ``idle`` 之一。
    """
    if channel_values.get("aborted"):
        return "aborted"
    if has_interrupt:
        return "interrupted"
    # 明确标记 done（部分任务 channel 里写了 status=done）
    if channel_values.get("status") == "done":
        return "done"
    # 成片已落库 → 完成
    if channel_values.get("final_movie_path") or channel_values.get(
        "final_movie_with_audio"
    ):
        return "done"
    # 全部镜头已生成且 perfect（无中断残留）→ 视为已完成。
    # 注意：next_nodes 非空但无 interrupt 通常是历史脏数据（如 run3 残留 branch:to:director），
    # 不能据此误判成 running，否则历史页会显示“仍在运行”。
    if all_scenes_done:
        return "done"
    if next_nodes:
        return "running"
    return "idle"


def _project_root() -> str:
    # history_api.py 位于 <root>/src/agent/multimedia/，需上溯 4 层到仓库根
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def _local_clip_path(idx: int) -> str:
    """本地片段绝对路径（output/clips/scene_{idx:02d}.mp4）。"""
    return os.path.join(_project_root(), "output", "clips", f"scene_{idx:02d}.mp4")


def _local_shot_path(idx: int) -> str:
    """审核片段绝对路径（output/review/shot_{idx+1:02d}.mp4）。"""
    return os.path.join(_project_root(), "output", "review", f"shot_{idx + 1:02d}.mp4")


def _local_final_movie_path() -> str:
    """最终成片绝对路径（output/review/FINAL_成片.mp4）。"""
    return os.path.join(_project_root(), "output", "review", "FINAL_成片.mp4")


def _is_remote(url: Any) -> bool:
    return isinstance(url, str) and url.startswith("http")


def _to_media_url(path: Any) -> str | None:
    """把本地绝对路径的产物转成可经 `/media` 访问的相对 URL；非本地路径返回 None。

    兼容两种情况：
      * 绝对路径含 ``<root>/output/...``：转成 ``/media/...``；
      * 已是 ``/media/...`` 或 http(s) 链接：原样返回。
    其余（如 base64 data URI / 空值）返回 None。
    """
    if not isinstance(path, str) or not path:
        return None
    path = path.replace("\\", "/")
    if path.startswith("/media/") or path.startswith("http"):
        return path
    if "data:" in path:
        return None
    root = _project_root().replace("\\", "/")
    out = root + "/output/"
    if path.startswith(out):
        return "/media/" + path[len(out):]
    # 兜底：按 /output/ 分隔符切分
    marker = "/output/"
    idx = path.find(marker)
    if idx != -1:
        return "/media/" + path[idx + len(marker):]
    return None


def _scan_thread_clips(thread_id: str) -> dict[int, str]:
    """扫描 output/clips/<thread_id>/ 下本次流水线的真实分镜产物。

    返回 {scene_index(0-based): "/media/clips/<thread_id>/agnes_vid_<ts>.mp4"}。
    文件名含毫秒时间戳（agnes_vid_<ts>.mp4），按时间升序对应第 0..N-1 镜。
    仅做单次请求内的轻量扫描（os.listdir + 正则），不写库。
    """
    if not thread_id:
        return {}
    d = os.path.join(_project_root(), "output", "clips", thread_id)
    if not os.path.isdir(d):
        return {}
    pat = re.compile(r"^agnes_vid_(\d+)\.mp4$")
    hits: list[tuple[int, str]] = []
    try:
        for fn in os.listdir(d):
            m = pat.match(fn)
            if m:
                hits.append((int(m.group(1)), fn))
    except OSError:
        return {}
    hits.sort(key=lambda x: x[0])  # 时间戳升序 → 第 0..N-1 镜
    return {idx: f"/media/clips/{thread_id}/{fn}" for idx, (_, fn) in enumerate(hits)}


def _best_local_video_media(
    i: int, thread_id: str = "", thread_clips: dict[int, str] | None = None
) -> str | None:
    """返回第 i 个镜头（0-based）的最佳本地可播视频 /media 路径；无则返回 None。

    优先级：
      1. output/clips/<thread_id>/agnes_vid_*.mp4（本次 thread 隔离真实产物）；
      2. output/clips/scene_{i:02d}.mp4（历史兼容 stitcher 缓存）；
      3. output/review/shot_{i+1:02d}.mp4（历史兼容审核片段）。
    """
    if thread_clips is None and thread_id:
        thread_clips = _scan_thread_clips(thread_id)
    if thread_clips and i in thread_clips:
        return thread_clips[i]
    clip = _local_clip_path(i)
    if os.path.isfile(clip):
        return f"/media/clips/scene_{i:02d}.mp4"
    shot = _local_shot_path(i)
    if os.path.isfile(shot):
        return f"/media/review/shot_{i + 1:02d}.mp4"
    return None


def _localize_history_urls(
    values: dict, thread_id: str = "", thread_clips: dict[int, str] | None = None
) -> None:
    """把往期会话里过期/无效的远程视频 URL 改写为本地 /media 路径（就地修改 values）。

    仅当本地文件真实存在时才改写，否则保留原值（前端会自行处理无效链接）。
    覆盖：每个镜头的 final_video_url/raw_video_url，以及图片 first_frame_url/image_url。
    优先用 output/clips/<thread_id>/agnes_vid_*.mp4 真实产物（按 thread 隔离扫描）。
    不触碰数据库，只影响当前响应。
    """
    if thread_clips is None:
        thread_clips = _scan_thread_clips(thread_id) if thread_id else {}
    scenes = values.get("scenes")
    if isinstance(scenes, list):
        for i, sc in enumerate(scenes):
            if not isinstance(sc, dict):
                continue
            local_media = _best_local_video_media(i, thread_id=thread_id, thread_clips=thread_clips)
            if local_media:
                # 优先用本线程隔离目录的真实产物（agnes_vid_*.mp4）。
                # 无条件覆盖：旧产物命名 scene_xx.mp4/shot_xx.mp4 在本线程目录中不存在，
                # 直接覆盖可避免前端 404（加载失败）；远程链接同理改写。
                for field in ("final_video_url", "raw_video_url", "video_url"):
                    if _is_remote(sc.get(field)) or not sc.get(field) or sc.get(field) != local_media:
                        sc[field] = local_media
            # 图片同样本地化（output/keyframes 或 output/clips/thread_id 下的首帧）
            for field in ("first_frame_url", "image_url"):
                cur = sc.get(field)
                if cur:
                    m = _to_media_url(cur)
                    if m:
                        sc[field] = m


async def _extract_executed_nodes(saver: AsyncSqliteSaver, cfg: dict) -> List[str]:
    """提取该 thread 已执行过的节点序列（按执行顺序去重）。

    LangGraph 每跑完一个节点会写一个 checkpoint，其 metadata 形如
    {"source": "loop", "step": n, "writes": {"<node_name>": {...}}}。
    按时间升序遍历即可还原节点执行轨迹，供前端重建进度时间轴。

    注意 alist 返回顺序为「最新→最旧」，需反转后再处理。
    """
    nodes: List[str] = []
    seen = set()
    try:
        history = [t async for t in saver.alist(cfg, limit=_MAX_MERGE_VERSIONS)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("提取已执行节点时 alist 失败: %s", type(exc).__name__)
        return nodes

    history.reverse()  # 旧 → 新
    for tup in history:
        meta = getattr(tup, "metadata", None) or {}
        writes = meta.get("writes")
        if not isinstance(writes, dict):
            continue
        for node_name in writes.keys():
            if (
                isinstance(node_name, str)
                and node_name
                and not node_name.startswith("__")
                and node_name not in seen
            ):
                seen.add(node_name)
                nodes.append(node_name)
    return nodes


async def _build_merged_values(saver: AsyncSqliteSaver, cfg: dict, latest: Any) -> Any:
    """构建“合并产物视图”的 channel_values，用于历史展示。

    部分往期会话的**最新/最完整 checkpoint** 里 scenes 的视频/图片 URL 已被后续
    resume/clear 清空，而**其他历史版本**中留有有效产物（DashScope 远程链接或本地路径）。
    这里遍历该 thread 全部 checkpoint 版本：
      * 以 scenes 数量最多的版本为骨架（保留脚本/critique 等元数据）；
      * 对每个 scene idx，从所有版本里收集最佳视频 URL 与图片 URL 回填；
      * 视频优先取本地 clips/review 文件，其次历史远程 http 链接。
    返回合并后的 channel_values dict；任何异常时回退为 latest 的 channel_values。
    不修改数据库，仅用于读取展示。
    """
    try:
        # 只回读最近 _MAX_MERGE_VERSIONS 个版本，避免对 1GB 库全量遍历。
        # alist 是异步生成器，必须用 async 推导式消费，不能 `await alist(...)`
        # （否则抛 "async_generator object can't be awaited"，导致合并永远失败，
        # 回退到 latest 后历史会话可能因 scenes 被 clear 而显示为空/不完整）。
        history = [t async for t in saver.alist(cfg, limit=_MAX_MERGE_VERSIONS)]
    except Exception as exc:  # noqa: BLE001 - alist 失败不影响主流程
        logger.warning("合并历史产物视图时 alist 失败，回退最新版本: %s", type(exc).__name__)
        return (latest.checkpoint or {}).get("channel_values") or {} if latest else {}

    if not history:
        return (latest.checkpoint or {}).get("channel_values") or {} if latest else {}

    def _scene_count(tup: Any) -> int:
        cv = (tup.checkpoint or {}).get("channel_values") or {}
        scenes = cv.get("scenes")
        return len(scenes) if isinstance(scenes, list) else 0

    # 【修复】骨架优先取“最新”版本的 scenes：
    # 早期以 scenes 最多版本为骨架，会把 run3 这类“最新版本已带产物、旧版是 8 个
    # 空镜头”的会话误判成旧骨架，导致视频/图片引用全丢。这里若最新版本 scenes
    # 非空则直接用它；仅当最新版本被 clear 为空时才回退到 scenes 最多的版本。
    def _latest_cv(tup: Any) -> dict:
        return (tup.checkpoint or {}).get("channel_values") or {}

    latest_scenes = _latest_cv(latest).get("scenes") if latest else None
    if isinstance(latest_scenes, list) and latest_scenes:
        base = dict(_latest_cv(latest))
    else:
        best = max(history, key=_scene_count)
        base = dict(_latest_cv(best))
    scenes = base.get("scenes")
    if not isinstance(scenes, list):
        return base

    # 【修复】从所有版本收集每个 idx 的最佳视频/图片 URL（首次命中即采用）。
    # 兼容三类来源：本地绝对路径（转 /media 相对 URL）、http 远程链接、/media 相对路径；
    # 跳过 base64 data URI（内嵌大图）与空值。此前只认 http 链接，导致本地产物
    # （如 run3 的 output/clips/.../agnes_vid_*.mp4）永远无法回填。
    vid_map: dict[int, str] = {}
    img_map: dict[int, str] = {}
    for tup in history:
        cv = _latest_cv(tup)
        for i, s in enumerate(cv.get("scenes") or []):
            if not isinstance(s, dict):
                continue
            if i not in vid_map:
                m = _to_media_url(s.get("final_video_url")) or _to_media_url(s.get("raw_video_url"))
                if m:
                    vid_map[i] = m
            if i not in img_map:
                im = s.get("first_frame_url") or s.get("image_url") or ""
                m = _to_media_url(im)
                if m:
                    img_map[i] = m

    # 回填到骨架 scenes
    for i, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            continue
        if i in vid_map:
            sc["final_video_url"] = vid_map[i]
            sc["raw_video_url"] = vid_map[i]
            sc.setdefault("video_url", vid_map[i])
        if i in img_map:
            sc["image_url"] = img_map[i]
            sc.setdefault("first_frame_url", img_map[i])

    # 顶层最终成片：严格按 thread_id 隔离，杜绝跨任务“张冠李戴”。
    # 规则（优先级从高到低）：
    #   1) 该任务自己 checkpoint 记录的 final_movie_path（含 final_movie_with_audio）；
    #   2) 该任务自己的隔离目录 output/clips/<thread_id>/FINAL_成片_subs_audio.mp4
    #      （主成片，带字幕+音轨） / FINAL_成片_subs.mp4（字幕） / FINAL_成片.mp4；
    #   3) 否则 None（前端不显示成片，而不是借用别的任务的通用文件）。
    # 【严禁】使用通用 output/review/FINAL_成片.mp4 兜底——那是旧运行产物，串任务。
    thread_id = (cfg or {}).get("configurable", {}).get("thread_id", "") or ""
    final_url: str | None = None
    fmp = base.get("final_movie_path")
    if isinstance(fmp, str) and fmp:
        final_url = _to_media_url(fmp)
    if final_url is None:
        fa = base.get("final_movie_with_audio")
        if isinstance(fa, str) and fa:
            final_url = _to_media_url(fa)
    if final_url is None and thread_id:
        own_dir = os.path.join(_project_root(), "output", "clips", thread_id)
        # 成片命名三变体统一兜底（名称优先级：带音轨 > 字幕 > 纯成片）
        for name in ("FINAL_成片_subs_audio.mp4", "FINAL_成片_subs.mp4", "FINAL_成片.mp4"):
            own_final = os.path.join(own_dir, name)
            if os.path.isfile(own_final):
                final_url = "/media/clips/" + thread_id + "/" + name
                break
    base["final_movie_path"] = final_url

    # 【修复】顶层“非 scene”业务字段（reference_sheets / assets_imported /
    # character_portrait_url 等）必须从**最新**版本优先回填：
    # 历史合并以 scenes 最多的旧版本为骨架，但素材注入（POST /assets/apply）
    # 写回的是最新版本——旧骨架里没有这些字段。若只返回骨架，前端读到的
    # reference_sheets 会为空（虽然 aget_tuple 直读最新版本非空）。这里用最新
    # 版本覆盖顶层业务字段（scene 级字段已由上面的回填逻辑处理，互不冲突）。
    latest_cv = (latest.checkpoint or {}).get("channel_values") or {} if latest else {}
    for key in ("reference_sheets", "assets_imported", "character_portrait_url",
                "thread_id", "task"):
        if key in latest_cv and latest_cv[key] not in (None, "", {}, []):
            base[key] = latest_cv[key]

    # 【修复】若 checkpoint 里 scenes 被 resume/clear 清空（finalize 后常见），但
    # 磁盘上 output/clips/<thread_id>/agnes_vid_*.mp4 真实存在，则按 thread 隔离目录
    # 扫描结果重建 scenes 列表，确保详情页能渲染真实分镜、不再“加载失败”。
    if thread_id and not isinstance(base.get("scenes"), list):
        base["scenes"] = []
    if thread_id and len(base.get("scenes") or []) == 0:
        clips = _scan_thread_clips(thread_id)
        if clips:
            base["scenes"] = [
                {
                    "index": i,
                    "final_video_url": url,
                    "raw_video_url": url,
                    "video_url": url,
                }
                for i, url in clips.items()
            ]
    return base




def _summarize(thread_id: str, values: dict, tup: Any = None) -> dict:
    """把 channel_values 压缩成列表项摘要（不含 scenes 全量）。

    Args:
        thread_id: 会话 ID。
        values: 合并产物视图的 channel_values（含完整 scenes）。
        tup: 最新 checkpoint tuple，用于提取中断信息与元数据；可为 None。
    """
    checkpoint = (tup.checkpoint or {}) if tup is not None else {}
    interrupt = _extract_interrupt(tup.pending_writes) if tup is not None else None
    next_nodes = _extract_next_nodes(values)

    task = values.get("task")
    title = None
    if isinstance(task, str) and task.strip():
        cleaned = " ".join(task.split())
        title = cleaned[:_TITLE_MAX]

    scenes = values.get("scenes")
    total_scenes = len(scenes) if isinstance(scenes, list) else 0

    raw_index = values.get("current_scene_index")
    current_scene_index = raw_index if isinstance(raw_index, int) else 0

    # 全部镜头均生成完成（video_is_perfect）且无空镜头 → 视为已完成，避免历史脏
    # next 节点（如 run3 残留 branch:to:director）导致误判 running。
    all_scenes_done = (
        total_scenes > 0
        and isinstance(scenes, list)
        and all(
            isinstance(s, dict)
            and (s.get("video_is_perfect") is True or s.get("image_url") or s.get("final_video_url") or s.get("raw_video_url"))
            for s in scenes
        )
    )

    status = _derive_status(values, interrupt is not None, next_nodes, all_scenes_done)

    # 【修复】历史任务若是终态（done/interrupted/aborted），但 current_scene_index
    # 仍停在初始 0（脚本跑完未回写），会导致前端进度显示成「1/X 镜头」——看着像
    # 只做了一点点，实际已全部完成。这里对终态把进度归一到末镜头，让列表/详情
    # 一致地呈现「全部完成」。仅影响展示，不写库。
    if total_scenes > 0 and current_scene_index < total_scenes - 1 and status in (
        "done",
        "interrupted",
        "aborted",
    ):
        current_scene_index = total_scenes - 1

    # 封面：首镜首帧（优先 first_frame_url/image_url 本地化，其次用隔离目录真实产物）。
    # 缺失则为空串，前端安全渲染（不报错）。
    cover = ""
    if isinstance(scenes, list) and scenes:
        first = scenes[0] if isinstance(scenes[0], dict) else {}
        for fld in ("first_frame_url", "image_url"):
            cv = first.get(fld)
            if cv:
                m = _to_media_url(cv)
                if m:
                    cover = m
                    break
    if not cover:
        clips = _scan_thread_clips(thread_id)
        if 0 in clips:
            cover = clips[0]

    return {
        "thread_id": thread_id,
        "task": title,
        "updated_at": checkpoint.get("ts"),
        "total_scenes": total_scenes,
        "current_scene_index": current_scene_index,
        "status": status,
        "has_interrupt": interrupt is not None,
        "next": next_nodes,
        "checkpoint_id": checkpoint.get("id"),
        "cover": cover,
    }


def _corrupted_summary(thread_id: str, reason: str) -> dict:
    """构造一条损坏占位摘要，保证列表整体可用。"""
    return {
        "thread_id": thread_id,
        "task": None,
        "updated_at": None,
        "total_scenes": 0,
        "current_scene_index": 0,
        "status": "corrupted",
        "has_interrupt": False,
        "next": [],
        "checkpoint_id": None,
        "error": reason,
    }


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@router.get("/threads")
async def list_history_threads(
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """分页返回历史会话摘要。

    只回传标题、进度与状态徽标，**不含** scenes 全量（含大量 URL 与 critique
    文本，全量返回会让响应膨胀到 MB 级）。完整数据请调用详情接口。

    Args:
        limit: 单页条数，1~200，默认 30。
        offset: 偏移量。

    Returns:
        ``{"threads": [...], "total": int, "limit": int, "offset": int}``
    """
    def _query() -> tuple[int, list[str]]:
        # sqlite3 连接不能跨线程使用，因此连接的创建、查询与关闭必须全部
        # 发生在同一个 to_thread 工作线程内。
        conn = _sync_ro_connect()
        try:
            total = conn.execute(
                "SELECT COUNT(DISTINCT thread_id) FROM checkpoints WHERE checkpoint_ns=''"
            ).fetchone()[0]
            # 走 sqlite_autoindex_checkpoints_1(thread_id, checkpoint_ns, checkpoint_id)，
            # 只扫索引不读 BLOB。checkpoint_id 是 UUIDv6，字典序即时间序。
            rows = conn.execute(
                "SELECT thread_id FROM checkpoints WHERE checkpoint_ns='' "
                "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return total, [r[0] for r in rows]
        finally:
            conn.close()

    try:
        total, thread_ids = await asyncio.to_thread(_query)
    except sqlite3.Error as exc:
        logger.error("历史会话列表查询失败: %s", exc)
        raise HTTPException(
            status_code=503, detail=f"历史记录数据库不可读: {exc}"
        ) from exc

    threads: list[dict] = []
    async with _SaverSession() as saver:
        for thread_id in thread_ids:
            try:
                # 单条解析加超时保护：个别 thread 的 checkpoint blob 过大时，
                # 同步反序列化会长时间阻塞 uvicorn 事件循环，拖垮整个列表响应。
                # 用 wait_for 隔离，避免一个坏 thread 让所有历史会话都加载不出来。
                tup = await asyncio.wait_for(
                    saver.aget_tuple({"configurable": {"thread_id": thread_id}}),
                    timeout=20,
                )
                if tup is None:
                    threads.append(_corrupted_summary(thread_id, "checkpoint 缺失"))
                    continue
                # 列表接口只展示标题/进度/状态徽标，**不需要**跨历史版本回填产物
                # （那是很重的 alist(limit=1000) 全量反序列化，见 _build_merged_values）。
                # 标题、镜头数、状态都来自最新版本的 channel_values，对列表展示足够准确；
                # 跨版本还原有效视频/图片 URL 的代价留给详情接口（get_history_thread），
                # 它只处理单个 thread，开销可控。否则多 thread 串行 alist(1000) 会让
                # 整个列表响应超过前端 20s 超时，触发 "The operation was aborted." 并降级。
                latest_values = (tup.checkpoint or {}).get("channel_values") or {}
                threads.append(_summarize(thread_id, latest_values, tup))
            except asyncio.TimeoutError:
                logger.warning("历史会话摘要解析超时（跳过）thread_id=%s", thread_id)
                threads.append(_corrupted_summary(thread_id, "解析超时"))
            except Exception as exc:  # noqa: BLE001 - 逐条隔离，不拖垮整个列表
                # 只记 thread_id 与异常类型，不打印 channel_values（含大量 URL）
                logger.warning(
                    "历史会话摘要解析失败 thread_id=%s error=%s",
                    thread_id,
                    type(exc).__name__,
                )
                threads.append(_corrupted_summary(thread_id, type(exc).__name__))

    # 排序：按 updated_at 时间倒序（最新在前）。checkpoints 表的 checkpoint_id
    # 混用了 UUIDv6（时间戳前缀，字典序≈时间序）与随机 UUIDv4，直接
    # ``ORDER BY MAX(checkpoint_id) DESC`` 会让旧 thread 因 UUIDv4 前缀而乱序排到
    # 最前（如 jimeng_*/01a0020f）。改用真实 updated_at（ISO8601）排序，并对
    # 脏数据（ts 误存为 UUID 字符串 / None）做兜底——无法解析的时间戳沉底，
    # 保证带真实时间戳的会话（如本次新跑的成片）稳定排在最前、可被用户看到。
    from datetime import datetime as _dt

    def _sort_key(t: dict):
        raw = t.get("updated_at")
        if isinstance(raw, str):
            try:
                ts = _dt.fromisoformat(raw.replace("Z", "+00:00"))
                return (0, -ts.timestamp())
            except ValueError:
                # 不可解析（脏 UUID 字符串 / 非 ISO 文本）→ 沉底
                return (1, 0)
        return (1, 0)

    threads.sort(key=_sort_key)

    return {
        "threads": threads,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/threads/{thread_id}")
async def get_history_thread(thread_id: str) -> dict[str, Any]:
    """返回单个历史会话的完整中间产物、进度与中断信息。

    Args:
        thread_id: 会话 ID。注意历史数据中并非全是 UUID（存在 ``story_1_*``
            这类脚本生成的 ID），因此**不做** UUID 校验。

    Returns:
        ``{"thread_id", "checkpoint_id", "values", "progress", "interrupt", "status"}``

    Raises:
        HTTPException: 会话不存在（404）或读取失败（500/503）。
    """
    if not thread_id or len(thread_id) > 200:
        raise HTTPException(status_code=400, detail="无效的 thread_id")

    latest_tup = None
    merged_values: dict = {}
    try:
        async with _SaverSession() as saver:
            tup = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
            if tup is not None:
                latest_tup = tup
                # 合并该 thread 所有历史版本里的有效产物（视频/图片），
                # 即使最新 checkpoint 已被 resume/clear 清空，也能还原生成过的内容。
                merged_values = await _build_merged_values(
                    saver, {"configurable": {"thread_id": thread_id}}, tup
                )
    except sqlite3.Error as exc:
        logger.error("历史会话库不可读 thread_id=%s: %s", thread_id, exc)
        raise HTTPException(
            status_code=503, detail=f"历史记录数据库不可读: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "读取历史会话失败 thread_id=%s error=%s",
            thread_id,
            type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"读取历史会话失败: {type(exc).__name__}"
        ) from exc

    if latest_tup is None:
        raise HTTPException(
            status_code=404, detail=f"未找到该历史会话的存档: {thread_id}"
        )

    checkpoint = latest_tup.checkpoint or {}
    raw_values = merged_values if merged_values else (checkpoint.get("channel_values") or {})

    interrupt = _extract_interrupt(latest_tup.pending_writes)
    next_nodes = _extract_next_nodes(raw_values)

    # 剔除框架内部通道（branch:to:* 等），只回传业务字段
    values = {
        key: _json_safe(val)
        for key, val in raw_values.items()
        if not key.startswith("branch:to:") and not key.startswith("__")
    }

    # 【修复】往期会话里 final_video_url/raw_video_url/final_movie_path 多为 DashScope
    # 临时远程链接（有时效/防盗链），前端直连会报“没有找到支持的视频格式和 mime 类型”。
    # 这里在**只读响应层**做 URL 本地化：若本地 output/clips（或 output/review）对应文件
    # 存在，则改写为本地 /media 路径，前端无需改动即可正常播放。注意：不写数据库。
    _localize_history_urls(values, thread_id=thread_id)

    scenes = raw_values.get("scenes")
    total_scenes = len(scenes) if isinstance(scenes, list) else 0
    raw_index = raw_values.get("current_scene_index")
    current_scene_index = raw_index if isinstance(raw_index, int) else 0

    all_scenes_done = (
        total_scenes > 0
        and isinstance(scenes, list)
        and all(
            isinstance(s, dict)
            and (s.get("video_is_perfect") is True or s.get("image_url") or s.get("final_video_url") or s.get("raw_video_url"))
            for s in scenes
        )
    )

    status = _derive_status(raw_values, interrupt is not None, next_nodes, all_scenes_done)

    # 与列表接口一致：终态任务把进度归一到末镜头
    if total_scenes > 0 and current_scene_index < total_scenes - 1 and status in (
        "done",
        "interrupted",
        "aborted",
    ):
        current_scene_index = total_scenes - 1

    # 【修复】历史会话丢失进度时间轴：
    # 前端 loadThread 后只能拿到 next 节点，无法还原"已跑过哪些节点"，
    # 导致点开历史任务进度条全空。这里从该 thread 全部 checkpoint 的
    # metadata.writes 中按 step 升序提取已执行节点序列，供前端重建时间轴。
    executed_nodes: List[str] = []
    try:
        async with _SaverSession() as saver:
            executed_nodes = await _extract_executed_nodes(
                saver, {"configurable": {"thread_id": thread_id}}
            )
    except Exception as exc:  # noqa: BLE001
        # 进度时间轴是展示增强，失败不得影响主数据返回
        logger.warning(
            "提取已执行节点失败 thread_id=%s: %s", thread_id, exc
        )

    return {
        "thread_id": thread_id,
        "checkpoint_id": checkpoint.get("id"),
        "updated_at": checkpoint.get("ts"),
        "values": values,
        "progress": {
            "next": next_nodes,
            "executed_nodes": executed_nodes,
            "current_scene_index": current_scene_index,
            "total_scenes": total_scenes,
            "step": (latest_tup.metadata or {}).get("step"),
        },
        "interrupt": interrupt,
        "status": status,
    }
