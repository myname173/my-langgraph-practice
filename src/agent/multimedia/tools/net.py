# -*- coding: utf-8 -*-
"""统一下载助手（P0-3 后续 · 真实链路修复）
==========================================
背景：本机存在代理（Clash 等，fake-IP 模式）。即梦 CDN（``*.byteimg.com``）
用 python ``requests`` 下载极不稳定：部分节点直连挂死（ReadTimeout 到超时），
部分节点连代理也超时；而系统 ``curl`` 稳定可用（直连/代理皆可）。

回退顺序（body 全部经 stdout 捕获，不落任何临时文件）：
  1. ``curl.exe`` 直连
  2. ``curl.exe`` 经本地代理
  3. ``requests`` 直连（短超时）
  4. ``requests`` 经本地代理

代理来源：``MULTIMEDIA_DOWNLOAD_PROXY`` > ``HTTPS_PROXY``/``https_proxy`` 等 > 端口探测。
可用 ``MULTIMEDIA_DOWNLOAD_PREFER=requests`` 切回 requests 优先。
调用方无需关心细节，直接拿 bytes / Response。
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
from typing import Optional

import requests

_PROXY_PORTS = (7897, 7890, 7891, 1087, 1080, 8889, 2080, 20171)
_proxy_cache: Optional[str] = None
_proxy_resolved = False

# requests 单次尝试的最长等待（避免部分 CDN 节点挂到调用方传来的大超时）
_REQUESTS_ATTEMPT_CAP = 45


def _detect_proxy() -> str:
    """解析可用代理地址；返回 '' 表示无代理。结果进程内缓存。"""
    global _proxy_cache, _proxy_resolved
    if _proxy_resolved:
        return _proxy_cache or ""
    _proxy_resolved = True
    for var in ("MULTIMEDIA_DOWNLOAD_PROXY", "HTTPS_PROXY", "https_proxy",
                "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        val = (os.getenv(var) or "").strip()
        if val:
            _proxy_cache = val
            return val
    for port in _PROXY_PORTS:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.25)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    _proxy_cache = f"http://127.0.0.1:{port}"
                    return _proxy_cache
        except Exception:  # noqa: BLE001
            continue
    _proxy_cache = ""
    return ""


# ── curl（body 走 stdout，无临时文件）──────────────────────
def _curl_exe() -> Optional[str]:
    exe = shutil.which("curl") or shutil.which("curl.exe")
    if exe:
        return exe
    for cand in (r"C:\Windows\System32\curl.exe", "/usr/bin/curl", "/usr/local/bin/curl"):
        if os.path.exists(cand):
            return cand
    return None


def _curl_fetch(url: str, timeout: int, headers: dict | None = None) -> bytes:
    """用 curl 下载并返回 bytes；先直连再经代理。"""
    exe = _curl_exe()
    if not exe:
        raise RuntimeError("curl 不可用")

    def _run(via_proxy: bool) -> bytes:
        _connect_to = int(min(12, max(3, timeout)))
        cmd = [exe, "-sS", "-L", "--fail",
               "--connect-timeout", str(_connect_to),
               "--max-time", str(int(timeout)),
               "--retry", "1", "--retry-delay", "1"]
        if via_proxy:
            proxy = _detect_proxy()
            if not proxy:
                raise RuntimeError("无可用代理")
            cmd += ["-x", proxy]
        if headers:
            for k, v in headers.items():
                cmd += ["-H", f"{k}: {v}"]
        cmd += [url]
        r = subprocess.run(cmd, capture_output=True, timeout=timeout + 20)
        if r.returncode != 0 or not r.stdout:
            err = (r.stderr or b"")[:200].decode("utf-8", "ignore")
            raise RuntimeError(f"curl rc={r.returncode} err={err}")
        return r.stdout

    try:
        return _run(via_proxy=False)
    except Exception:
        return _run(via_proxy=True)


class _CurlResponse:
    """最小 Response 兼容层（供 stream / .content 消费方使用）。"""
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200
        self.headers = {"Content-Length": str(len(content))}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int = 1 << 20):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


# ── 计划构造 ───────────────────────────────────────────────
def _bytes_plan(timeout: int):
    """生成 (kind, arg) 计划；kind ∈ {'curl','req'}。"""
    prefer_req = (os.getenv("MULTIMEDIA_DOWNLOAD_PREFER", "") or "").strip().lower() == "requests"
    proxy = _detect_proxy()
    curl_plans = [("curl", False)]
    if proxy:
        curl_plans.append(("curl", True))
    req_plans = [("req", None)]
    if proxy:
        req_plans.append(("req", {"http": proxy, "https": proxy}))
    return (req_plans + curl_plans) if prefer_req else (curl_plans + req_plans)


def _req_get(url, headers, timeout, proxies, session=None, stream=False):
    getter = (session or requests).get
    return getter(url, headers=headers, timeout=min(int(timeout), _REQUESTS_ATTEMPT_CAP),
                  proxies=proxies, stream=stream)


# ── 对外 API ───────────────────────────────────────────────
def fetch_bytes(url: str, timeout: int = 120, headers: dict | None = None,
                session: Optional[requests.Session] = None, label: str = "下载") -> bytes:
    """GET 返回 bytes（curl 优先，requests 兜底）。"""
    last_err: Exception | None = None
    for kind, arg in _bytes_plan(timeout):
        try:
            if kind == "curl":
                return _curl_fetch(url, timeout, headers)
            r = _req_get(url, headers, timeout, arg, session=session, stream=False)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last_err = e
    assert last_err is not None
    raise last_err


def fetch_response(url: str, timeout: int = 120, headers: dict | None = None,
                   stream: bool = False, session: Optional[requests.Session] = None,
                   attempts_per_plan: int = 1, label: str = "下载"):
    """GET 返回 Response（已 raise_for_status）。stream 兜底时返回兼容层。"""
    last_err: Exception | None = None
    for kind, arg in _bytes_plan(timeout):
        for _ in range(max(1, attempts_per_plan)):
            try:
                if kind == "curl":
                    return _CurlResponse(_curl_fetch(url, timeout, headers))
                r = _req_get(url, headers, timeout, arg, session=session, stream=stream)
                r.raise_for_status()
                return r
            except Exception as e:  # noqa: BLE001
                last_err = e
    assert last_err is not None
    raise last_err
