"""统一的 ``/media`` URL 与本地路径互转。

本项目历史上存在两种 ``/media`` 约定，导致跨模块解析不一致：

* **A) output-relative（正统）**：``/media/keyframes/x.png`` -> ``<root>/output/keyframes/x.png``。
  这是 ``static_server`` 的文档约定，也是 agnes/jimeng/embedding 早期按 output/ 拼接的前提。
* **B) root-relative（历史遗留）**：``/media/output/keyframes/x.png`` -> ``<root>/output/keyframes/x.png``。
  由 ``reference_gen._local_media_path`` 早期实现产出（直接 relative_to(PROJECT_ROOT)），
  会让按 A 约定解析的消费者拼出 ``output/output/...``（即「即梦参考图转 base64 失败」根因），
  也会让视觉模型拿到无法解析的 URL（即 400 code 1210）。

本模块统一按「A 为产出约定、两种都认」处理：
* :func:`local_to_media_url` 一律产出 A 约定（output/ 内为 ``/media/<rel>``；
  output/ 之外退化为带项目根名的 ``/media/<root>/<rel>``，由 static_server 的 root_marker 兜底）。
* :func:`media_url_to_local` 解析时对 A / B / 绝对路径形式都做尝试，返回真实存在的本地文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# tools/media_paths.py -> tools -> multimedia -> agent -> src -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_DIR = _PROJECT_ROOT / "output"
MEDIA_PREFIX = "/media/"


def project_root() -> Path:
    return _PROJECT_ROOT


def output_dir() -> Path:
    return _OUTPUT_DIR


def media_url_to_local(url: "str | None") -> Optional[Path]:
    """把 ``/media/<...>`` URL 解析为本地文件绝对路径；解析不到（或不存在）返回 None。

    对以下形式都做尝试（取第一个真实存在的文件）：
      1. 含项目根名的绝对路径形式：``/media/<root>/<rel>``；
      2. root-relative（遗留 B）：``/media/output/<rel>``；
      3. output-relative（正统 A）：``/media/<rel>``；
      4. 兜底：``/media/<rel>`` 直接相对项目根。
    """
    if not url or not isinstance(url, str) or not url.startswith(MEDIA_PREFIX):
        return None
    rel = url[len(MEDIA_PREFIX):].replace("\\", "/")
    if not rel:
        return None
    root_marker = f"{_PROJECT_ROOT.name}/"
    candidates = []
    if root_marker in rel:
        candidates.append(_PROJECT_ROOT / rel.split(root_marker, 1)[1])
    if rel.startswith("output/"):
        candidates.append(_PROJECT_ROOT / rel)
    candidates.append(_OUTPUT_DIR / rel)
    candidates.append(_PROJECT_ROOT / rel)
    for cand in candidates:
        try:
            if cand.is_file():
                return cand.resolve()
        except OSError:
            continue
    return None


def local_to_media_url(path: "str | Path | None") -> Optional[str]:
    """把项目内的本地路径转为 ``/media`` URL（统一产出 output-relative 约定）。

    * 位于 ``<root>/output/`` 之内 -> ``/media/<相对 output 的路径>``；
    * 位于项目内但在 output/ 之外 -> ``/media/<项目根名>/<相对项目根的路径>``
      （static_server 的 root_marker 兜底可解析）；
    * 不在项目内 / 无法解析 -> None。
    """
    if not path:
        return None
    try:
        p = Path(path).resolve()
        try:
            rel = p.relative_to(_OUTPUT_DIR)
            return MEDIA_PREFIX + str(rel).replace("\\", "/")
        except ValueError:
            pass
        rel = p.relative_to(_PROJECT_ROOT)
        return MEDIA_PREFIX + _PROJECT_ROOT.name + "/" + str(rel).replace("\\", "/")
    except (ValueError, OSError):
        return None
