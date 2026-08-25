"""Pi 会话状态的持久化存储。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_THINKING = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


class StateCorruptError(RuntimeError):
    """会话状态文件损坏，需要隔离后重建。"""


@dataclass(slots=True)
class SessionRecord:
    """一个 AstrBot 会话对应的 Pi 会话设置。"""

    session_id: str
    provider: str
    model: str
    thinking: str
    created_at: str


class SessionStateStore:
    """以原子替换方式保存会话映射，避免写入中断损坏主文件。"""

    def __init__(
        self,
        path: Path,
        default_provider: str,
        default_model: str,
        default_thinking: str,
    ) -> None:
        self.path = path
        self.default_provider = default_provider
        self.default_model = default_model
        self.default_thinking = _coerce_thinking(default_thinking, "high")
        self._lock = asyncio.Lock()
        self._sessions: dict[str, SessionRecord] = {}
        self.last_warning = ""

    async def initialize(self) -> None:
        """读取已有状态；损坏时隔离并重建，不存在时创建空状态文件。"""
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                await asyncio.to_thread(self._write_payload)
                return
            try:
                payload = await asyncio.to_thread(self._read_payload)
                self._sessions = self._decode_sessions(payload)
            except (StateCorruptError, TypeError, ValueError) as exc:
                backup = await asyncio.to_thread(self._quarantine)
                self._sessions = {}
                self.last_warning = (
                    f"会话状态已损坏，已隔离为 {backup.name} 并重建：{exc}"
                )
                await asyncio.to_thread(self._write_payload)

    @staticmethod
    def conversation_key(unified_msg_origin: str) -> str:
        """将平台会话标识转换为稳定且适合作为目录名的键。"""
        return hashlib.sha256(unified_msg_origin.encode("utf-8")).hexdigest()[:32]

    async def get_or_create(self, unified_msg_origin: str) -> SessionRecord:
        key = self.conversation_key(unified_msg_origin)
        async with self._lock:
            record = self._sessions.get(key)
            if record is None:
                record = self._new_record()
                self._sessions[key] = record
                await asyncio.to_thread(self._write_payload)
            return record

    async def rotate(self, unified_msg_origin: str) -> SessionRecord:
        """创建新的 Pi 会话，旧会话文件保留用于审计或手工恢复。"""
        key = self.conversation_key(unified_msg_origin)
        async with self._lock:
            previous = self._sessions.get(key)
            record = self._new_record(
                provider=previous.provider if previous else None,
                model=previous.model if previous else None,
                thinking=previous.thinking if previous else None,
            )
            self._sessions[key] = record
            await asyncio.to_thread(self._write_payload)
            return record

    async def set_model(
        self,
        unified_msg_origin: str,
        provider: str,
        model: str,
    ) -> SessionRecord:
        key = self.conversation_key(unified_msg_origin)
        async with self._lock:
            record = self._sessions.get(key) or self._new_record()
            record.provider = provider
            record.model = model
            self._sessions[key] = record
            await asyncio.to_thread(self._write_payload)
            return record

    async def set_thinking(
        self,
        unified_msg_origin: str,
        thinking: str,
    ) -> SessionRecord:
        key = self.conversation_key(unified_msg_origin)
        normalized = _coerce_thinking(thinking, "")
        if not normalized:
            raise ValueError(f"无效思考等级：{thinking}")
        async with self._lock:
            record = self._sessions.get(key) or self._new_record()
            record.thinking = normalized
            self._sessions[key] = record
            await asyncio.to_thread(self._write_payload)
            return record

    def _new_record(
        self,
        provider: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
    ) -> SessionRecord:
        return SessionRecord(
            session_id=str(uuid.uuid4()),
            provider=provider or self.default_provider,
            model=model or self.default_model,
            thinking=_coerce_thinking(thinking, self.default_thinking),
            created_at=datetime.now(UTC).isoformat(),
        )

    def _read_payload(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Pi 会话状态文件读取失败：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise StateCorruptError(f"Pi 会话状态不是合法 JSON：{exc}") from exc
        if not isinstance(raw, dict):
            raise StateCorruptError("Pi 会话状态文件的根节点必须是对象")
        return raw

    def _decode_sessions(self, payload: dict[str, Any]) -> dict[str, SessionRecord]:
        sessions = payload.get("sessions", {})
        if not isinstance(sessions, dict):
            raise StateCorruptError("Pi 会话状态文件中的 sessions 必须是对象")
        decoded: dict[str, SessionRecord] = {}
        for key, value in sessions.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            try:
                thinking = _coerce_thinking(
                    str(value["thinking"]), self.default_thinking
                )
                decoded[key] = SessionRecord(
                    session_id=str(value["session_id"]),
                    provider=str(value["provider"]),
                    model=str(value["model"]),
                    thinking=thinking,
                    created_at=str(value["created_at"]),
                )
            except KeyError:
                continue
        return decoded

    def _write_payload(self) -> None:
        payload = {
            "version": 1,
            "sessions": {
                key: asdict(record) for key, record in sorted(self._sessions.items())
            },
        }
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _quarantine(self) -> Path:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        backup = self.path.with_name(f"{self.path.name}.corrupt-{timestamp}")
        sequence = 1
        while backup.exists():
            backup = self.path.with_name(
                f"{self.path.name}.corrupt-{timestamp}-{sequence}"
            )
            sequence += 1
        self.path.replace(backup)
        return backup


def _coerce_thinking(value: str | None, default: str) -> str:
    text = str(value or "").strip().lower()
    if text in VALID_THINKING:
        return text
    return default
