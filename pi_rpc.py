"""通过 Pi 官方 JSONL/RPC 协议执行任务。"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import shutil
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state_store import SessionRecord, SessionStateStore

ProgressCallback = Callable[["ToolProgress"], Awaitable[None] | None]

_DIALOG_UI_METHODS = {"select", "confirm", "input", "editor"}
_TOOL_DETAIL_KEYS = (
    "path",
    "file_path",
    "file",
    "command",
    "pattern",
    "query",
    "glob",
    "url",
)


class PiError(RuntimeError):
    """Pi 集成的基础异常。"""


class PiBusyError(PiError):
    """同一会话已有任务运行。"""


class PiNotRunningError(PiError):
    """当前会话没有可停止的任务。"""


class PiAbortedError(PiError):
    """任务被用户或插件终止。"""


class PiTimeoutError(PiError):
    """任务超过允许的执行时间。"""


class PiProtocolError(PiError):
    """Pi 输出不符合 JSONL/RPC 协议。"""


@dataclass(frozen=True, slots=True)
class PiSettings:
    command: str
    config_dir: Path
    session_dir: Path
    workspace_root: Path
    provider: str
    model: str
    thinking: str = "high"
    tools: tuple[str, ...] = ("read", "bash", "edit", "write", "grep", "find", "ls")
    timeout_seconds: float = 600
    abort_grace_seconds: float = 8
    stats_timeout_seconds: float = 5
    command_args: tuple[str, ...] = ()
    approve_project_files: bool = True


@dataclass(frozen=True, slots=True)
class ToolProgress:
    name: str
    finished: bool
    failed: bool = False
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PiRunResult:
    text: str
    tools: tuple[ToolProgress, ...]
    stats: dict[str, Any] = field(default_factory=dict)


def split_text(text: str, max_chars: int) -> list[str]:
    """优先在换行处分段，并保证每段都不超过平台限制。"""
    normalized = text.strip()
    if not normalized:
        return []
    limit = max(200, max_chars)
    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > limit:
        boundary = remaining.rfind("\n", 0, limit + 1)
        if boundary < limit // 3:
            boundary = remaining.rfind("。", 0, limit + 1)
            if boundary >= limit // 3:
                boundary += 1
        if boundary < limit // 3:
            boundary = limit
        chunks.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def normalize_rpc_image(raw: str) -> dict[str, str] | None:
    """把 AstrBot 图片载荷整理为 Pi RPC 所需的 image 对象。"""
    text = str(raw or "").strip()
    if not text:
        return None
    mime = "image/png"
    if text.startswith("data:") and "," in text:
        header, text = text.split(",", 1)
        declared = header.split(":", 1)[1].split(";", 1)[0].strip()
        if declared:
            mime = declared
    elif text.startswith("base64://"):
        text = text.removeprefix("base64://")
        mime = _sniff_image_mime(text)
    else:
        mime = _sniff_image_mime(text)
    text = "".join(text.split())
    if not text:
        return None
    return {"type": "image", "data": text, "mimeType": mime}


def _sniff_image_mime(b64_text: str) -> str:
    try:
        padding = "=" * ((4 - len(b64_text[:64]) % 4) % 4)
        head = base64.b64decode(b64_text[:64] + padding, validate=False)
    except (ValueError, TypeError):
        return "image/png"
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if head.startswith(b"GIF8"):
        return "image/gif"
    if head.startswith(b"RIFF") and b"WEBP" in head[:16]:
        return "image/webp"
    return "image/png"


class PiRunner:
    """管理按 AstrBot 会话隔离的 Pi RPC 任务。"""

    def __init__(self, settings: PiSettings) -> None:
        self.settings = settings
        self._active: dict[str, _PiTurn] = {}
        self._active_lock = asyncio.Lock()

    async def check_runtime(self) -> str:
        """验证 Pi 可执行文件，并返回版本号。"""
        invocation = self._resolve_invocation()
        environment = self._environment()
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *invocation,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
                **self._windows_process_options(),
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        except FileNotFoundError as exc:
            raise PiError(f"找不到 Pi 可执行文件：{self.settings.command}") from exc
        except TimeoutError as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise PiError("Pi 版本检查超时") from exc
        if process.returncode != 0:
            detail = _redact(stderr.decode("utf-8", errors="replace").strip())
            raise PiError(f"Pi 版本检查失败：{detail or process.returncode}")
        version = stdout.decode("utf-8", errors="replace").strip()
        if not version:
            raise PiError("Pi 版本检查没有返回版本号")
        return version

    async def run_prompt(
        self,
        unified_msg_origin: str,
        record: SessionRecord,
        prompt: str,
        progress: ProgressCallback | None = None,
        images: Sequence[dict[str, str]] | None = None,
    ) -> PiRunResult:
        normalized_images = _clean_images(images)
        text = prompt.strip()
        if not text and not normalized_images:
            raise PiError("任务内容不能为空")
        if not text:
            text = "请查看这些图片"
        key = SessionStateStore.conversation_key(unified_msg_origin)
        turn = self._create_turn(key, record)
        await self._register(key, turn)
        try:
            return await turn.run_prompt(text, progress, normalized_images)
        finally:
            await turn.close()
            await self._unregister(key, turn)

    async def steer(
        self,
        unified_msg_origin: str,
        prompt: str,
        images: Sequence[dict[str, str]] | None = None,
    ) -> None:
        """把新指令转入当前仍在运行的任务，而不是再开一个进程。"""
        normalized_images = _clean_images(images)
        text = prompt.strip()
        if not text and not normalized_images:
            raise PiError("转向内容不能为空")
        if not text:
            text = "请查看这些图片"
        key = SessionStateStore.conversation_key(unified_msg_origin)
        async with self._active_lock:
            turn = self._active.get(key)
        if turn is None:
            raise PiNotRunningError("当前会话没有运行中的 Pi 任务")
        await turn.steer(text, normalized_images)

    async def set_thinking_active(self, unified_msg_origin: str, level: str) -> bool:
        """向运行中的进程发送思考等级；没有活动任务时返回 False。"""
        key = SessionStateStore.conversation_key(unified_msg_origin)
        async with self._active_lock:
            turn = self._active.get(key)
        if turn is None:
            return False
        await turn.send_control({"type": "set_thinking_level", "level": level})
        return True

    async def get_models(
        self,
        unified_msg_origin: str,
        record: SessionRecord,
    ) -> list[dict[str, Any]]:
        response = await self._run_command(
            unified_msg_origin,
            record,
            {"type": "get_available_models"},
        )
        models = response.get("models", [])
        return [item for item in models if isinstance(item, dict)]

    async def get_thinking_levels(
        self,
        unified_msg_origin: str,
        record: SessionRecord,
    ) -> list[str]:
        response = await self._run_command(
            unified_msg_origin,
            record,
            {"type": "get_available_thinking_levels"},
        )
        levels = response.get("levels", [])
        return [str(item) for item in levels if str(item).strip()]

    async def get_stats(
        self,
        unified_msg_origin: str,
        record: SessionRecord,
    ) -> dict[str, Any]:
        return await self._run_command(
            unified_msg_origin,
            record,
            {"type": "get_session_stats"},
        )

    async def compact(
        self,
        unified_msg_origin: str,
        record: SessionRecord,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        command: dict[str, Any] = {"type": "compact"}
        if instruction:
            command["customInstructions"] = instruction
        return await self._run_command(unified_msg_origin, record, command)

    async def is_active(self, unified_msg_origin: str) -> bool:
        key = SessionStateStore.conversation_key(unified_msg_origin)
        async with self._active_lock:
            return key in self._active

    async def stop(self, unified_msg_origin: str) -> None:
        key = SessionStateStore.conversation_key(unified_msg_origin)
        async with self._active_lock:
            turn = self._active.get(key)
        if turn is None:
            raise PiNotRunningError("当前会话没有运行中的 Pi 任务")
        await turn.abort()

    async def stop_all(self) -> None:
        async with self._active_lock:
            turns = list(self._active.values())
        await asyncio.gather(*(turn.abort() for turn in turns), return_exceptions=True)
        await asyncio.gather(*(turn.close() for turn in turns), return_exceptions=True)

    async def _run_command(
        self,
        unified_msg_origin: str,
        record: SessionRecord,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        key = SessionStateStore.conversation_key(unified_msg_origin)
        turn = self._create_turn(key, record)
        await self._register(key, turn)
        try:
            return await turn.run_command(command)
        finally:
            await turn.close()
            await self._unregister(key, turn)

    def _create_turn(self, key: str, record: SessionRecord) -> _PiTurn:
        workspace = self.settings.workspace_root / key
        return _PiTurn(
            settings=self.settings,
            invocation=self._resolve_invocation(),
            environment=self._environment(),
            workspace=workspace,
            record=record,
            name=f"AstrBot-{key[:8]}",
        )

    async def _register(self, key: str, turn: _PiTurn) -> None:
        async with self._active_lock:
            if key in self._active:
                raise PiBusyError("当前会话已有 Pi 任务运行，请等待完成或发送 /pi停止")
            self._active[key] = turn

    async def _unregister(self, key: str, turn: _PiTurn) -> None:
        async with self._active_lock:
            if self._active.get(key) is turn:
                self._active.pop(key, None)

    def _resolve_invocation(self) -> tuple[str, ...]:
        configured = self.settings.command.strip()
        candidates = [configured] if configured else []
        candidates.extend(["pi.cmd", "pi"] if os.name == "nt" else ["pi"])
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            resolved = str(path) if path.exists() else shutil.which(candidate)
            if not resolved:
                continue
            if os.name == "nt" and resolved.lower().endswith(".ps1"):
                shell = shutil.which("pwsh") or shutil.which("powershell")
                if shell:
                    return (
                        shell,
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        resolved,
                        *self.settings.command_args,
                    )
            return (resolved, *self.settings.command_args)
        raise PiError(f"找不到 Pi 可执行文件：{configured or 'pi'}")

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PI_CODING_AGENT_DIR"] = str(self.settings.config_dir)
        environment["PI_TELEMETRY"] = "0"
        return environment

    @staticmethod
    def _windows_process_options() -> dict[str, Any]:
        if os.name != "nt":
            return {}
        return {"creationflags": 0x08000000}


class _PiTurn:
    """单次 Pi RPC 进程及其协议状态。"""

    def __init__(
        self,
        settings: PiSettings,
        invocation: Sequence[str],
        environment: dict[str, str],
        workspace: Path,
        record: SessionRecord,
        name: str,
    ) -> None:
        self.settings = settings
        self.invocation = tuple(invocation)
        self.environment = environment
        self.workspace = workspace
        self.record = record
        self.name = name
        self.process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._kill_task: asyncio.Task[None] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=40)
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._started = asyncio.Event()
        self._abort_requested = False
        self._seq = 0

    async def run_prompt(
        self,
        prompt: str,
        progress: ProgressCallback | None,
        images: Sequence[dict[str, str]] | None = None,
    ) -> PiRunResult:
        await self._start()
        request_id = self._next_id("prompt")
        payload: dict[str, Any] = {
            "id": request_id,
            "type": "prompt",
            "message": prompt,
        }
        if images:
            payload["images"] = list(images)
        await self._send(payload)
        if self._abort_requested:
            await self.abort()
        try:
            result = await asyncio.wait_for(
                self._collect_prompt(request_id, progress),
                timeout=self.settings.timeout_seconds,
            )
        except TimeoutError as exc:
            await self.abort()
            raise PiTimeoutError(
                f"Pi 任务超过 {self.settings.timeout_seconds:g} 秒，已停止"
            ) from exc
        if result.stats or self._abort_requested:
            return result
        stats = await self._try_stats()
        return PiRunResult(text=result.text, tools=result.tools, stats=stats)

    async def run_command(self, command: dict[str, Any]) -> dict[str, Any]:
        await self._start()
        request = dict(command)
        request_id = self._next_id("command")
        request["id"] = request_id
        await self._send(request)
        try:
            return await asyncio.wait_for(
                self._wait_response(request_id),
                timeout=min(self.settings.timeout_seconds, 60),
            )
        except TimeoutError as exc:
            raise PiTimeoutError("Pi 控制命令等待响应超时") from exc

    async def steer(
        self,
        prompt: str,
        images: Sequence[dict[str, str]] | None = None,
    ) -> None:
        await self._wait_until_running()
        payload: dict[str, Any] = {
            "id": self._next_id("steer"),
            "type": "steer",
            "message": prompt,
        }
        if images:
            payload["images"] = list(images)
        await self._send(payload)

    async def send_control(self, command: dict[str, Any]) -> None:
        await self._wait_until_running()
        request = dict(command)
        request["id"] = self._next_id("control")
        await self._send(request)

    async def _wait_until_running(self) -> None:
        await self._started.wait()
        if (
            self._abort_requested
            or self.process is None
            or self.process.returncode is not None
        ):
            raise PiNotRunningError("当前 Pi 任务正在停止或已结束")

    async def abort(self) -> None:
        self._abort_requested = True
        process = self.process
        if process is None or process.returncode is not None:
            return
        try:
            await self._send({"id": self._next_id("abort"), "type": "abort"})
        except (BrokenPipeError, ConnectionError, PiError):
            pass
        if self._kill_task is None or self._kill_task.done():
            self._kill_task = asyncio.create_task(self._force_kill_later(process))

    async def close(self) -> None:
        async with self._close_lock:
            if self._kill_task is not None:
                self._kill_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._kill_task
                self._kill_task = None
            process = self.process
            if process is None:
                return
            if process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
                try:
                    await process.stdin.wait_closed()
                except (AttributeError, BrokenPipeError, ConnectionError):
                    pass
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
            if self._stderr_task is not None:
                await asyncio.gather(self._stderr_task, return_exceptions=True)
            self.process = None

    async def _force_kill_later(self, process: asyncio.subprocess.Process) -> None:
        await asyncio.sleep(max(0.05, self.settings.abort_grace_seconds))
        if self.process is process and process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()

    async def _start(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.settings.config_dir.mkdir(parents=True, exist_ok=True)
        self.settings.session_dir.mkdir(parents=True, exist_ok=True)
        arguments = [
            *self.invocation,
            "--mode",
            "rpc",
            "--session-id",
            self.record.session_id,
            "--session-dir",
            str(self.settings.session_dir),
            "--name",
            self.name,
            "--provider",
            self.record.provider,
            "--model",
            self.record.model,
            "--thinking",
            self.record.thinking,
        ]
        if self.settings.tools:
            arguments.extend(("--tools", ",".join(self.settings.tools)))
        if self.settings.approve_project_files:
            arguments.append("--approve")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *arguments,
                cwd=str(self.workspace),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.environment,
                limit=1024 * 1024,
                **PiRunner._windows_process_options(),
            )
        except (FileNotFoundError, OSError) as exc:
            self._started.set()
            raise PiError(f"Pi 进程启动失败：{exc}") from exc
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._started.set()

    async def _collect_prompt(
        self,
        request_id: str,
        progress: ProgressCallback | None,
    ) -> PiRunResult:
        deltas: list[str] = []
        final_text = ""
        tools: list[ToolProgress] = []
        error_message = ""
        acknowledged = False
        while True:
            event = await self._read_event()
            event_type = event.get("type")
            if event_type == "response" and event.get("id") == request_id:
                if not event.get("success", False):
                    raise PiError(str(event.get("error") or "Pi 拒绝了任务"))
                acknowledged = True
            elif event_type == "message_update":
                update = event.get("assistantMessageEvent", {})
                if update.get("type") == "text_delta":
                    deltas.append(str(update.get("delta", "")))
            elif event_type == "message_end":
                message = event.get("message", {})
                if message.get("role") == "assistant":
                    authoritative = _assistant_text(message)
                    if authoritative:
                        final_text = authoritative
                    if message.get("errorMessage"):
                        error_message = str(message["errorMessage"])
            elif event_type == "tool_execution_start":
                item = ToolProgress(
                    str(event.get("toolName", "未知工具")),
                    False,
                    detail=_tool_detail(event),
                )
                tools.append(item)
                await _notify(progress, item)
            elif event_type == "tool_execution_end":
                item = ToolProgress(
                    str(event.get("toolName", "未知工具")),
                    True,
                    bool(event.get("isError", False)),
                    _tool_detail(event),
                )
                tools.append(item)
                await _notify(progress, item)
            elif event_type == "extension_ui_request":
                await self._handle_extension_ui(event)
            elif event_type == "agent_settled":
                break

        text = final_text or "".join(deltas).strip()
        if self._abort_requested:
            if text:
                return PiRunResult(
                    text=f"（任务已停止）\n{text}",
                    tools=tuple(tools),
                    stats={},
                )
            raise PiAbortedError("Pi 任务已停止")
        if not acknowledged:
            raise PiProtocolError("Pi 未确认 prompt 请求")
        if not text and error_message:
            raise PiError(f"Pi 执行失败：{_redact(error_message)}")
        if not text:
            raise PiError("Pi 已结束任务，但没有返回文本结果")
        return PiRunResult(text=text, tools=tuple(tools), stats={})

    async def _try_stats(self) -> dict[str, Any]:
        if (
            self._abort_requested
            or self.process is None
            or self.process.returncode is not None
        ):
            return {}
        try:
            await self._send(
                {"id": self._next_id("stats"), "type": "get_session_stats"}
            )
            return await asyncio.wait_for(
                self._wait_response_prefix("stats-"),
                timeout=self.settings.stats_timeout_seconds,
            )
        except (PiError, TimeoutError):
            return {}

    async def _handle_extension_ui(self, event: dict[str, Any]) -> None:
        method = str(event.get("method") or "")
        request_id = event.get("id")
        if method not in _DIALOG_UI_METHODS or not request_id:
            return
        await self._send(
            {
                "type": "extension_ui_response",
                "id": request_id,
                "cancelled": True,
            }
        )

    async def _wait_response(self, request_id: str) -> dict[str, Any]:
        return await self._wait_response_prefix(request_id, exact=True)

    async def _wait_response_prefix(
        self,
        request_id: str,
        exact: bool = False,
    ) -> dict[str, Any]:
        while True:
            event = await self._read_event()
            if event.get("type") == "extension_ui_request":
                await self._handle_extension_ui(event)
                continue
            if event.get("type") != "response":
                continue
            event_id = str(event.get("id") or "")
            matched = (
                event_id == request_id if exact else event_id.startswith(request_id)
            )
            if not matched:
                continue
            if not event.get("success", False):
                raise PiError(str(event.get("error") or "Pi 控制命令失败"))
            data = event.get("data")
            return data if isinstance(data, dict) else {}

    async def _send(self, payload: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            raise PiError("Pi 进程未运行")
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            process.stdin.write(encoded)
            await process.stdin.drain()

    async def _read_event(self) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise PiError("Pi 标准输出不可用")
        while True:
            try:
                raw = await process.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError) as exc:
                raise PiProtocolError("Pi 返回了超长 JSONL 行") from exc
            if not raw:
                await process.wait()
                if self._abort_requested:
                    raise PiAbortedError("Pi 任务已停止")
                detail = _redact("\n".join(self._stderr_lines))
                suffix = f"：{detail[-1500:]}" if detail else ""
                raise PiError(f"Pi 进程异常退出（代码 {process.returncode}）{suffix}")
            if raw.endswith(b"\n"):
                raw = raw[:-1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            if not raw.strip():
                continue
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PiProtocolError("Pi 返回了无法解析的 JSONL 数据") from exc
            if not isinstance(event, dict):
                raise PiProtocolError("Pi RPC 事件必须是 JSON 对象")
            return event

    async def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        while True:
            raw = await process.stderr.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                self._stderr_lines.append(line)

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"


async def _notify(callback: ProgressCallback | None, item: ToolProgress) -> None:
    if callback is None:
        return
    result = callback(item)
    if inspect.isawaitable(result):
        await result


def _clean_images(images: Sequence[dict[str, str]] | None) -> list[dict[str, str]]:
    if not images:
        return []
    cleaned: list[dict[str, str]] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        data = str(item.get("data") or "").strip()
        if not data:
            continue
        mime = str(item.get("mimeType") or "image/png").strip() or "image/png"
        cleaned.append({"type": "image", "data": data, "mimeType": mime})
    return cleaned


def _tool_detail(event: dict[str, Any]) -> str:
    args = event.get("args")
    if not isinstance(args, dict):
        return ""
    for key in _TOOL_DETAIL_KEYS:
        value = args.get(key)
        if value is None:
            continue
        text = " ".join(str(value).split())
        if not text:
            continue
        return text[:77] + "..." if len(text) > 80 else text
    return ""


def _assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts).strip()


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def _redact(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1<已隐藏>", redacted)
        else:
            redacted = pattern.sub("<已隐藏>", redacted)
    return redacted
