"""AstrBot 插件入口与命令行为测试。"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

from astrbot_plugin_pi.pi_rpc import PiRunResult


class FakeFilter:
    @staticmethod
    def command(*_args, **_kwargs):
        return lambda function: function

    @staticmethod
    def permission_type(*_args, **_kwargs):
        return lambda function: function


class FakeMessageChain:
    def __init__(self) -> None:
        self.text = ""

    def message(self, text: str):
        self.text = text
        return self


class FakeImage:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    async def convert_to_base64(self) -> str:
        return self.payload


class FakeEvent:
    def __init__(
        self,
        umo: str = "aiocqhttp:FriendMessage:10001",
        images: list[FakeImage] | None = None,
    ) -> None:
        self.unified_msg_origin = umo
        self.sent: list[str] = []
        self.call_llm = True
        self._images = images or []

    def plain_result(self, text: str) -> str:
        return text

    async def send(self, chain: FakeMessageChain) -> None:
        self.sent.append(chain.text)

    def should_call_llm(self, enabled: bool) -> None:
        self.call_llm = enabled

    def get_messages(self) -> list[FakeImage]:
        return self._images


class FakeContext:
    pass


class FakeRunner:
    def __init__(self) -> None:
        self.active = False
        self.stopped = False
        self.steered = ""
        self.last_images: list[dict[str, str]] = []
        self.compacted = ""

    async def check_runtime(self) -> str:
        return "0.84.2-test"

    async def run_prompt(self, _umo, _record, prompt, progress=None, images=None):
        self.last_images = list(images or [])
        if progress:
            await progress(
                types.SimpleNamespace(
                    name="read",
                    finished=False,
                    failed=False,
                    detail="README.md",
                )
            )
        suffix = f"：{self.last_images[0]['mimeType']}" if self.last_images else ""
        return PiRunResult(text=f"结果：{prompt}{suffix}", tools=(), stats={})

    async def steer(self, _umo, prompt, images=None):
        self.steered = str(prompt)
        self.last_images = list(images or [])

    async def get_models(self, _umo, _record):
        return [
            {"provider": "91grok", "id": "grok-4.5"},
            {"provider": "openai", "id": "gpt-test"},
        ]

    async def get_thinking_levels(self, _umo, _record):
        return ["low", "medium", "high"]

    async def set_thinking_active(self, _umo, level):
        return bool(self.active)

    async def get_stats(self, _umo, _record):
        return {
            "userMessages": 3,
            "toolCalls": 2,
            "tokens": {"total": 99},
            "contextUsage": {"tokens": 60, "contextWindow": 200, "percent": 30},
        }

    async def compact(self, _umo, _record, instruction=None):
        self.compacted = str(instruction or "")
        return {
            "summary": "已压缩历史上下文",
            "tokensBefore": 100,
            "estimatedTokensAfter": 20,
        }

    async def is_active(self, _umo):
        return self.active

    async def stop(self, _umo):
        self.stopped = True

    async def stop_all(self):
        self.stopped = True


def _load_main(data_dir: Path):
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    event_filter = types.ModuleType("astrbot.api.event.filter")
    star = types.ModuleType("astrbot.api.star")
    core = types.ModuleType("astrbot.core")
    core_star = types.ModuleType("astrbot.core.star")
    core_filter = types.ModuleType("astrbot.core.star.filter")
    command = types.ModuleType("astrbot.core.star.filter.command")
    star_tools = types.ModuleType("astrbot.core.star.star_tools")

    class Logger:
        @staticmethod
        def info(_message: str) -> None:
            pass

        @staticmethod
        def warning(_message: str) -> None:
            pass

        @staticmethod
        def error(_message: str) -> None:
            pass

    class PermissionType:
        ADMIN = "admin"

    class Context:
        pass

    class Star:
        def __init__(self, context) -> None:
            self.context = context

    class StarTools:
        @staticmethod
        def get_data_dir(_name: str) -> Path:
            return data_dir

    def register(*_args, **_kwargs):
        return lambda plugin_class: plugin_class

    api.logger = Logger()
    event.AstrMessageEvent = FakeEvent
    event.MessageChain = FakeMessageChain
    event.filter = FakeFilter()
    event_filter.PermissionType = PermissionType
    star.Context = Context
    star.Star = Star
    star.register = register
    command.GreedyStr = str
    star_tools.StarTools = StarTools

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.event.filter": event_filter,
            "astrbot.api.star": star,
            "astrbot.core": core,
            "astrbot.core.star": core_star,
            "astrbot.core.star.filter": core_filter,
            "astrbot.core.star.filter.command": command,
            "astrbot.core.star.star_tools": star_tools,
        }
    )
    sys.modules.pop("astrbot_plugin_pi.main", None)
    return importlib.import_module("astrbot_plugin_pi.main")


async def collect(generator) -> list[str]:
    return [item async for item in generator]


def test_帮助和正常任务都会关闭默认LLM(tmp_path: Path) -> None:
    async def scenario() -> None:
        main = _load_main(tmp_path)
        plugin = main.PiControlPlugin(FakeContext(), {"show_tool_progress": False})
        plugin._runner = FakeRunner()
        await plugin.initialize()

        help_event = FakeEvent()
        help_output = await collect(plugin.cmd_pi(help_event, ""))
        task_event = FakeEvent()
        task_output = await collect(plugin.cmd_pi(task_event, "整理项目"))

        assert "/pi停止" in help_output[0]
        assert "/pi压缩" in help_output[0]
        assert help_event.call_llm is False
        assert "Pi 已接收任务" in task_output[0]
        assert task_output[-1] == "结果：整理项目"
        assert task_event.call_llm is False
        await plugin.terminate()

    asyncio.run(scenario())


def test_会话新建状态模型和思考等级(tmp_path: Path) -> None:
    async def scenario() -> None:
        main = _load_main(tmp_path)
        plugin = main.PiControlPlugin(FakeContext(), {})
        plugin._runner = FakeRunner()
        await plugin.initialize()
        event = FakeEvent()

        new_result = await plugin.cmd_new(event)
        status = await plugin.cmd_status(event)
        models = await plugin.cmd_model(event, "")
        changed_model = await plugin.cmd_model(event, "openai/gpt-test")
        changed_thinking = await plugin.cmd_thinking(event, "中")
        rejected_thinking = await plugin.cmd_thinking(event, "超级")
        unsupported = await plugin.cmd_thinking(event, "极高")
        compacted = await plugin.cmd_compact(event, "保留接口")

        assert "已创建新的 Pi 会话" in new_result
        assert "累计 Token：99" in status
        assert "上下文：30%" in status
        assert "openai/gpt-test" in models
        assert changed_model == "已切换 Pi 模型：openai/gpt-test"
        assert changed_thinking == "已将 Pi 思考等级设为：medium"
        assert "无效等级" in rejected_thinking
        assert "当前模型不支持 xhigh" in unsupported
        assert "Token：100 → 20" in compacted

    asyncio.run(scenario())


def test_活动任务会转向而不是拒绝(tmp_path: Path) -> None:
    async def scenario() -> None:
        main = _load_main(tmp_path)
        plugin = main.PiControlPlugin(FakeContext(), {})
        runner = FakeRunner()
        runner.active = True
        plugin._runner = runner
        await plugin.initialize()
        event = FakeEvent()

        output = await collect(plugin.cmd_pi(event, "改做别的"))
        thinking = await plugin.cmd_thinking(event, "低")

        assert output == ["已将新指令转入当前 Pi 任务。"]
        assert runner.steered == "改做别的"
        assert "已应用到当前任务" in thinking
        assert "先发送 /pi停止" in await plugin.cmd_new(event)
        assert "暂时不能切换模型" in await plugin.cmd_model(event, "grok-4.5")

    asyncio.run(scenario())


def test_消息链图片会传给Pi(tmp_path: Path) -> None:
    async def scenario() -> None:
        main = _load_main(tmp_path)
        plugin = main.PiControlPlugin(FakeContext(), {"show_tool_progress": False})
        runner = FakeRunner()
        plugin._runner = runner
        await plugin.initialize()
        event = FakeEvent(images=[FakeImage("base64://" + "iVBORw0KGgo")])
        output = await collect(plugin.cmd_pi(event, "看图"))

        assert runner.last_images
        assert runner.last_images[0]["mimeType"] == "image/png"
        assert "image/png" in output[-1]

    asyncio.run(scenario())


def test_配置模式是有效JSON且没有密钥字段() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
    assert schema["provider"]["default"] == "91grok"
    assert schema["thinking"]["options"] == [
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert "api_key" not in schema
