"""Pi RPC 进程管理与协议边界测试。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from astrbot_plugin_pi.pi_rpc import (
    PiAbortedError,
    PiBusyError,
    PiError,
    PiProtocolError,
    PiRunner,
    PiSettings,
    PiTimeoutError,
    normalize_rpc_image,
    split_text,
)
from astrbot_plugin_pi.state_store import SessionRecord, SessionStateStore

FAKE_PI = Path(__file__).with_name("fake_pi.py")
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
    "AAAABJRU5ErkJggg=="
)


def run(coroutine):
    return asyncio.run(coroutine)


def make_runner(tmp_path: Path, timeout: float = 2) -> PiRunner:
    return PiRunner(
        PiSettings(
            command=sys.executable,
            command_args=(str(FAKE_PI),),
            config_dir=tmp_path / "config",
            session_dir=tmp_path / "sessions",
            workspace_root=tmp_path / "workspaces",
            provider="91grok",
            model="grok-4.5",
            timeout_seconds=timeout,
            abort_grace_seconds=0.15,
            stats_timeout_seconds=0.2,
        )
    )


def make_record() -> SessionRecord:
    return SessionRecord(
        session_id="00000000-0000-4000-8000-000000000001",
        provider="91grok",
        model="grok-4.5",
        thinking="high",
        created_at="2026-08-21T00:00:00+00:00",
    )


def test_运行时检查和正常RPC任务(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path)
        progress = []

        assert await runner.check_runtime() == "0.84.2-fake"
        result = await runner.run_prompt(
            "aiocqhttp:FriendMessage:10001",
            make_record(),
            "读取项目",
            progress.append,
        )

        assert result.text == "测试完成：读取项目"
        assert [item.name for item in progress] == ["read", "read"]
        assert progress[0].detail == "README.md"
        assert result.stats["tokens"]["total"] == 42
        assert not await runner.is_active("aiocqhttp:FriendMessage:10001")

    run(scenario())


def test_模型列表使用结构化RPC响应(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path)
        models = await runner.get_models("会话一", make_record())
        assert [(item["provider"], item["id"]) for item in models] == [
            ("91grok", "grok-4.5"),
            ("openai", "gpt-test"),
        ]

    run(scenario())


def test_同一会话拒绝并发并支持停止(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path, timeout=5)
        first = asyncio.create_task(
            runner.run_prompt("同一会话", make_record(), "等待停止")
        )
        for _ in range(100):
            if await runner.is_active("同一会话"):
                break
            await asyncio.sleep(0.01)

        with pytest.raises(PiBusyError, match="已有 Pi 任务"):
            await runner.run_prompt("同一会话", make_record(), "第二个任务")
        await runner.stop("同一会话")
        with pytest.raises(PiAbortedError, match="已停止"):
            await first
        assert not await runner.is_active("同一会话")

    run(scenario())


def test_运行中转向会并入当前任务(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path, timeout=5)
        first = asyncio.create_task(
            runner.run_prompt("转向会话", make_record(), "等待停止")
        )
        for _ in range(100):
            if await runner.is_active("转向会话"):
                break
            await asyncio.sleep(0.01)

        await runner.steer("转向会话", "改做别的")
        result = await first
        assert result.text == "已转向：改做别的"
        assert not await runner.is_active("转向会话")

    run(scenario())


def test_忽略停止时会强制结束子进程(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path, timeout=5)
        first = asyncio.create_task(
            runner.run_prompt("强杀会话", make_record(), "忽略停止")
        )
        for _ in range(100):
            if await runner.is_active("强杀会话"):
                break
            await asyncio.sleep(0.01)

        await runner.stop("强杀会话")
        with pytest.raises(PiAbortedError, match="已停止"):
            await first
        assert not await runner.is_active("强杀会话")

    run(scenario())


def test_扩展对话框取消后任务能完成(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path)
        result = await runner.run_prompt("扩展会话", make_record(), "扩展确认")
        assert result.text == "测试完成：扩展确认"

    run(scenario())


def test_统计超时不会丢掉已完成正文(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path)
        result = await runner.run_prompt("统计会话", make_record(), "卡住统计")
        assert result.text == "统计已完成"
        assert result.stats == {}

    run(scenario())


def test_图片会随prompt发送(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path)
        result = await runner.run_prompt(
            "图片会话",
            make_record(),
            "看图",
            images=[{"type": "image", "data": PNG_B64, "mimeType": "image/png"}],
        )
        assert result.text == "测试完成：看图：image/png"

    run(scenario())


def test_压缩命令返回摘要(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path)
        result = await runner.compact("压缩会话", make_record(), "保留接口")
        assert result["summary"] == "已压缩历史上下文"
        assert result["tokensBefore"] == 100

    run(scenario())


def test_任务超时会结束子进程(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path, timeout=0.1)
        with pytest.raises(PiTimeoutError, match="已停止"):
            await runner.run_prompt("超时会话", make_record(), "等待超时")
        assert not await runner.is_active("超时会话")

    run(scenario())


def test_插件卸载可批量关闭多个活动会话(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path, timeout=5)
        tasks = [
            asyncio.create_task(runner.run_prompt(name, make_record(), "等待停止"))
            for name in ("会话甲", "会话乙")
        ]
        for _ in range(100):
            states = [await runner.is_active(name) for name in ("会话甲", "会话乙")]
            if all(states):
                break
            await asyncio.sleep(0.01)

        await runner.stop_all()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        assert all(isinstance(item, PiError) for item in outcomes)
        assert not await runner.is_active("会话甲")
        assert not await runner.is_active("会话乙")

    run(scenario())


def test_损坏JSONL会返回协议错误并清理进程(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path)
        with pytest.raises(PiProtocolError, match="JSONL"):
            await runner.run_prompt("损坏会话", make_record(), "损坏协议")
        assert not await runner.is_active("损坏会话")

    run(scenario())


def test_没有文本时返回明确错误(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = make_runner(tmp_path)
        with pytest.raises(PiError, match="没有返回文本"):
            await runner.run_prompt("空结果会话", make_record(), "没有文本")

    run(scenario())


def test_会话状态跨实例持久化并可轮换(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "state.json"
        first = SessionStateStore(path, "91grok", "grok-4.5", "high")
        await first.initialize()
        original = await first.get_or_create("平台:私聊:10001")
        await first.set_thinking("平台:私聊:10001", "xhigh")

        second = SessionStateStore(path, "其他", "其他模型", "low")
        await second.initialize()
        restored = await second.get_or_create("平台:私聊:10001")
        rotated = await second.rotate("平台:私聊:10001")

        assert restored.session_id == original.session_id
        assert restored.thinking == "xhigh"
        assert rotated.session_id != restored.session_id
        assert rotated.model == "grok-4.5"
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1

    run(scenario())


def test_损坏状态文件会隔离并重建(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "state.json"
        path.write_text("{not-json", encoding="utf-8")
        store = SessionStateStore(path, "91grok", "grok-4.5", "high")
        await store.initialize()
        record = await store.get_or_create("平台:私聊:10001")

        assert record.provider == "91grok"
        assert "已隔离" in store.last_warning
        backups = list(tmp_path.glob("state.json.corrupt-*"))
        assert len(backups) == 1
        assert json.loads(path.read_text(encoding="utf-8"))["sessions"]

    run(scenario())


def test_长文本分段覆盖换行句号和超长单词() -> None:
    text = "第一段。\n" + "甲" * 250 + "。" + "乙" * 250
    chunks = split_text(text, 200)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")
    assert all(1 <= len(chunk) <= 200 for chunk in chunks)


def test_图片载荷会推断MIME() -> None:
    item = normalize_rpc_image("base64://" + PNG_B64)
    assert item is not None
    assert item["type"] == "image"
    assert item["mimeType"] == "image/png"
    assert item["data"] == PNG_B64
