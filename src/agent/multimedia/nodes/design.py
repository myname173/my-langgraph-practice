# src/agent/multimedia/nodes/design.py
"""视觉设计阶段：参考图生成、资产匹配（角色 / 场景锚点基线）。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
from ..reference_gen import generate_reference_sheets
from ..state import MultimediaState
from .common import _resolve_style


def reference_gen_node(state: MultimediaState, config: dict | None = None):
    """
    Reference Sheet Generation Node
    ================================
    在关键帧生成之前，为角色（4视图转面图）、道具（多角度设定图）、
    场景环境（建立镜头）生成一致性参考图。

    架构位置: showrunner_review → 【reference_gen_node】→ visual_context_builder
    """
    print("--- [Reference] Generating consistency reference sheets ---")

    thread_id = (config or {}).get("configurable", {}).get("thread_id", "") if config else ""
    global_setting = state.get("global_setting", "")
    scenes = state.get("scenes", [])
    task = state.get("task", "")
    style = _resolve_style(state)
    style_key = style["style_key"]
    style_suffix = style["style_suffix"]

    # ── P2-5：跨集资产库复用 ──
    # 连续剧的第 N 集若携带 series_assets（存于 run 目录之外的剧集资产库），
    # 直接复用，跳过参考图生成——既省额度，又保证跨集的角色/场景外观一致。
    series_assets = state.get("series_assets") or {}
    if (series_assets.get("characters") or series_assets.get("props")
            or series_assets.get("environments")):
        episode = state.get("episode") or 1
        print(f"    [Asset] P2-5 跨集资产库命中（第 {episode} 集），复用 series_assets，跳过参考图生成")
        character_portrait_url = state.get("character_portrait_url")
        if not character_portrait_url and series_assets.get("characters"):
            first_char = next(iter(series_assets["characters"].values()))
            if first_char:
                character_portrait_url = _first_url(first_char.get("urls") or first_char.get("url"))
        report = _build_asset_match_baseline(series_assets, scenes)
        return {
            "reference_sheets": series_assets,
            "character_portrait_url": character_portrait_url,
            "thread_id": thread_id or state.get("thread_id"),
            "assets_imported": True,
            "asset_match_report": report,
        }

    # 跳过生成：用户已注入参考图资产（即梦历史图经 AssetLibraryPanel 标注保存为
    # reference_sheets）。直接复用现有 reference_sheets，省额度且保持角色一致。
    if state.get("assets_imported"):
        existing = state.get("reference_sheets") or {}
        if existing.get("characters") or existing.get("props") or existing.get("environments"):
            print("    [Asset] 检测到已注入素材，跳过参考图生成，直接复用 reference_sheets")
            character_portrait_url = state.get("character_portrait_url")
            if not character_portrait_url and existing.get("characters"):
                first_char = next(iter(existing["characters"].values()))
                if first_char:
                    character_portrait_url = _first_url(first_char.get("urls") or first_char.get("url"))
            # 基线：预扫描全剧本，标记哪些资产名从未被任何镜头脚本提及（静默失效预警）
            report = _build_asset_match_baseline(existing, scenes)
            return {
                "reference_sheets": existing,
                "character_portrait_url": character_portrait_url,
                "thread_id": thread_id or state.get("thread_id"),
                "assets_imported": True,
                "asset_match_report": report,
            }
        # 未注入素材时，若已存在 character_portrait_url（上一轮代理生成），直接复用，
        # 不再重新调用 jimeng 生成角色肖像（避免 jimeng 产物混入参考图素材）。
        if state.get("character_portrait_url"):
            print("    [Asset] 复用已有 character_portrait_url，跳过 jimeng 肖像生成")
            return {
                "reference_sheets": {"characters": {}, "props": {}, "environments": {}},
                "character_portrait_url": state["character_portrait_url"],
                "thread_id": thread_id or state.get("thread_id"),
            }

    try:
        sheets = generate_reference_sheets(
            global_setting=global_setting,
            scenes=scenes,
            task=task,
            style_suffix=style_suffix,
            style_key=style_key,
            thread_id=thread_id,
        )
    except Exception as e:
        print(f"    [WARN] Reference sheet generation failed (non-blocking): {e}")
        sheets = {"characters": {}, "props": {}, "environments": {}}

    # 从角色转面图中提取第一张作为 character_portrait_url（向后兼容）
    character_portrait_url = state.get("character_portrait_url")
    if sheets.get("characters"):
        first_char = next(iter(sheets["characters"].values()))
        if first_char:
            character_portrait_url = _first_url(first_char.get("url"))
            print(f"    [OK] character_portrait_url updated from turnaround sheet")

    # 统计
    n_chars = len(sheets.get("characters", {}))
    n_props = len(sheets.get("props", {}))
    n_envs = len(sheets.get("environments", {}))
    print(f"    [Reference] Generated: {n_chars} characters, {n_props} props, {n_envs} environments")

    # 基线：自动生成的参考图同样做一次命中预扫描，便于前端统一展示「是否参与生成」
    report = _build_asset_match_baseline(sheets, scenes)

    return {
        "reference_sheets": sheets,
        "character_portrait_url": character_portrait_url,
        "thread_id": thread_id or state.get("thread_id"),
        "asset_match_report": report,
    }


def _first_url(url):
    """把 reference entry 的 url 字段（string | list | None）归一为单字符串。

    用于向后兼容旧 checkpoint 中单 url 的资产，以及需要"首张"的场景
    （如源头角色肖像 character_portrait_url）。
    """
    if not url:
        return None
    if isinstance(url, list):
        for u in url:
            if u:
                return u
        return None
    return url


def _to_urls(url):
    """把 reference entry 的 url 字段（string | list | None）归一为字符串列表。

    一个资产名下多张参考图（如"女主"含正面/侧脸/全身）全部保留，
    供 _match_reference_elements 返回给 image_gen_node 做多参考图融合注入。
    """
    if not url:
        return []
    if isinstance(url, list):
        return [u for u in url if u]
    return [url]


# 命中则为"整张多视图网格图"（turntable / turnaround sheet / 三视图并排等），
# 仅作离线身份参考，禁止作为视频生成的帧内容注入（遵循 Veo/Runway/Seedance 规范）。
_REF_TURNAROUND_HINTS = (
    "turnaround", "turntable", "three-view", "three view", "3-view", "3 view",
    "model sheet", "reference sheet", "design sheet", "四视图", "三视图", "转面图",
    "多角度", "views arranged side by side", "multiple views", "side by side",
)


def _is_turnaround_sheet(entry: dict) -> bool:
    """判断一个 reference_sheets 条目是否为整张多视图网格图（需排除）。

    依据优先顺序：
      1. 显式 kind == "turnaround"（新规范）；
      2. description / url 文件名含 turnaround 等关键词（回退）。
    """
    if not isinstance(entry, dict):
        return False
    kind = entry.get("kind", "")
    if str(kind).lower() == "turnaround":
        return True
    hay = " ".join([
        str(entry.get("description", "")),
        str(entry.get("name_cn", "")),
        str(entry.get("url", "")),
        str(entry.get("urls", "")),
    ]).lower()
    return any(h in hay for h in _REF_TURNAROUND_HINTS)


def _filter_reference_images(
    ref_sheets: dict,
    portrait_url: str = "",
    max_refs: int = 2,
) -> tuple:
    """排除整张 turnaround/多视图网格图，仅保留角色肖像与道具/环境锚点，限制张数。

    返回：(filtered_url_list, notes_list)

    - characters 类别：仅保留「非 turnaround」的单视图肖像（description 不含网格关键词）；
      turnaround 整图一律剔除（避免网格残留）。
    - props / environments：保留（道具/环境单图，无多视图网格风险）。
    - 源头角色肖像 character_portrait_url 始终保留（最高优先身份锚点）。
    - 总张数限制为 max_refs（参考 Runway/Seedance「1~2 张参考 + 首尾帧」规范），
      超量时按「肖像 > 角色单视图 > 道具 > 环境」截断，抑制多图相互竞争导致的构图杂乱/身份漂移。
    """
    kept: list = []
    notes: list = []
    _cats_order = ("characters", "props", "environments")

    # 1) 源头肖像（最高优先）
    if portrait_url and portrait_url not in kept:
        kept.append(portrait_url)
        notes.append("[Character Consistency] 源头肖像：保持同一角色身份锚点。")

    # 2) 遍历各分类，剔除 turnaround 整图
    for _cat in _cats_order:
        for _name, _data in (ref_sheets.get(_cat) or {}).items():
            if isinstance(_data, dict):
                if _is_turnaround_sheet(_data):
                    # 整张多视图网格图：排除（不注入视频生成）
                    continue
                _raw = _data.get("urls")
                if _raw is None:
                    _raw = _data.get("url")
                for _u in _to_urls(_raw):
                    if _u and _u not in kept:
                        kept.append(_u)
                if _data.get("description"):
                    notes.append(f"[{_cat[:-1].title()} '{_name}']: {_data['description']}")
            elif isinstance(_data, str) and _data and _data not in kept:
                kept.append(_data)

    # 3) 截断到上限（肖像已在首位，优先保留）
    if len(kept) > max_refs:
        print(f"    [*] 参考图 {len(kept)} 张超过上限 {max_refs}，截断（肖像优先）")
        kept = kept[:max_refs]

    return kept, notes


def _all_asset_names(reference_sheets: dict) -> list:
    """收集 reference_sheets 全部资产名（角色/道具/环境），供基线报告遍历。"""
    names = []
    for cat in ("characters", "props", "environments"):
        for name in (reference_sheets.get(cat) or {}).keys():
            if name and name not in names:
                names.append(name)
    return names


def _build_asset_match_baseline(reference_sheets: dict, scenes: list) -> dict:
    """预扫描全剧本，建立素材匹配基线。

    返回 {"matched": {name: 0}, "unmatched": [name,...]}：
    - matched[name]=0 表示该资产名在剧本中至少出现一次（后续 image_gen_node 累加命中镜头数）；
    - unmatched 为在全部镜头脚本中都搜不到的资产名（即极可能「静默失效」）。
    这样前端无需知道剧本内容，也能提示用户「这张图没被用到」。
    """
    scripts = " ".join(
        str(s.get("script", "") or "") for s in (scenes or [])
    ).lower()
    matched, unmatched = {}, []
    for name in _all_asset_names(reference_sheets):
        # 复用通用匹配（name_cn + 单词级 fallback）
        data = None
        for cat in ("characters", "props", "environments"):
            data = (reference_sheets.get(cat) or {}).get(name)
            if data is not None:
                break
        hit = False
        name_cn = data.get("name_cn", "") if isinstance(data, dict) else ""
        names = [n for n in [name, name_cn] if n]
        if any(n.lower() in scripts for n in names):
            hit = True
        else:
            hit = any(
                word in scripts
                for n in names
                for word in n.lower().replace("_", " ").split()
                if len(word) > 1
            )
        if hit:
            matched[name] = 0
        else:
            unmatched.append(name)
    return {"matched": matched, "unmatched": unmatched}


def _match_reference_elements(reference_sheets: dict, script_text: str) -> dict:
    """匹配 reference sheets 中与当前剧本相关的角色/道具/环境。

    一个角色/环境资产可挂多张参考图，匹配命中后返回 **全部 url 列表**，
    供 image_gen_node 多参考图融合注入，强化角色/场景一致性（对齐 Vidu）。

    Returns:
        {
            "char_urls": list[str] or None,
            "char_desc": str,
            "prop_desc": str,
            "env_urls": list[str] or None,
            "matched_char_name": str,
            "matched_prop_name": str,
        }
    """
    result = {"char_urls": None, "char_desc": "", "prop_desc": "",
              "env_urls": None, "matched_char_name": "", "matched_prop_name": ""}

    def _check_match(name: str, data: dict) -> bool:
        """通用名称匹配：支持 name_cn + name + 单词级 fallback。

        额外增加 description 反向匹配：当用户上传素材以英文名（如 char_main）
        命名、而剧本 script 使用中文角色名时，name 无法命中，此时用素材
        description 中的中文特征词反向到 script 查找，避免所有镜头因匹配
        失败而回退到同一张 character_portrait_url，造成首帧雷同。
        """
        data = data if isinstance(data, dict) else {}
        name_cn = data.get("name_cn", "")
        names = [n for n in [name, name_cn] if n]
        if any(n.lower() in script_text for n in names):
            return True
        if any(
            word in script_text
            for n in names
            for word in n.lower().replace("_", " ").split()
            if len(word) > 1
        ):
            return True
        # description 反向匹配：素材描述里的特征词出现在剧本中
        desc = (data.get("description") or "").lower()
        if desc:
            # 取描述中的中文/实义词（长度>1）尝试命中剧本
            for _tok in desc.replace("_", " ").split():
                if len(_tok) > 1 and _tok in script_text:
                    return True
        return False

    # 角色匹配（支持多角色同时命中：合并所有命中角色的 url，供多参考图融合）
    _matched_char_urls = []
    _matched_char_desc = []
    for char_name, char_data in reference_sheets.get("characters", {}).items():
        if _check_match(char_name, char_data):
            _raw = char_data.get("urls") if isinstance(char_data, dict) else None
            if _raw is None and isinstance(char_data, dict):
                _raw = char_data.get("url")
            _u = _to_urls(_raw) if isinstance(char_data, dict) else _to_urls(char_data)
            if _u:
                _matched_char_urls.extend(_u)
                _matched_char_desc.append(char_data.get("description", "") if isinstance(char_data, dict) else "")
                if not result["matched_char_name"]:
                    result["matched_char_name"] = char_name
    if _matched_char_urls:
        result["char_urls"] = _matched_char_urls
        result["char_desc"] = "；".join([d for d in _matched_char_desc if d])

    # 道具匹配
    for prop_name, prop_data in reference_sheets.get("props", {}).items():
        if _check_match(prop_name, prop_data):
            result["prop_desc"] = prop_data.get("description", "") if isinstance(prop_data, dict) else ""
            result["matched_prop_name"] = prop_name
            break

    # 环境匹配
    for env_name, env_data in reference_sheets.get("environments", {}).items():
        if _check_match(env_name, env_data):
            _raw = env_data.get("urls") if isinstance(env_data, dict) else None
            if _raw is None and isinstance(env_data, dict):
                _raw = env_data.get("url")
            result["env_urls"] = _to_urls(_raw) if isinstance(env_data, dict) else _to_urls(env_data)
            break

    return result
