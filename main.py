"""AstrBot Pi 控制插件入口。"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import PermissionType
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.star.star_tools import StarTools

from .pi_rpc import (
    PiAbortedError,
    PiBusyError,
    PiError,
    PiNotRunningError,
    PiRunner,
    PiSettings,
    PiTimeoutError,
    ToolProgress,
    normalize_rpc_image,
    split_text,
)
from .state_store import VALID_THINKING, SessionStateStore

PLUGIN_NAME = "astrbot_plugin_pi"
THINKING_LEVELS = {
    "关": "off",
    "关闭": "off",
    "off": "off",
    "最小": "minimal",
    "minimal": "minimal",
    "低": "low",
    "low": "low",
    "中": "medium",
    "medium": "medium",
    "高": "high",
    "high": "high",
    "极高": "xhigh",
    "xhigh": "xhigh",
    "最大": "max",
    "max": "max",
}


@register(
    PLUGIN_NAME,
    "Codex",
    "通过 AstrBot 聊天控制 Pi 编码智能体",
    "v1.1.0",
)
class PiControlPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        config_dir = Path(self._cfg_str("pi_config_dir", str(data_dir / "pi_config")))
        settings = PiSettings(
            command=self._cfg_str(
                "pi_command", "/AstrBot/data/pi_runtime/node_modules/.bin/pi"
            ),
            config_dir=config_dir,
            session_dir=Path(self._cfg_str("session_dir", str(data_dir / "sessions"))),
            workspace_root=Path(
                self._cfg_str("workspace_root", "/AstrBot/data/pi_workspaces")
            ),
            provider=self._cfg_str("provider", "91grok"),
            model=self._cfg_str("model", "grok-4.5"),
            thinking=self._cfg_str("thinking", "high"),
            tools=tuple(
                item.strip()
                for item in self._cfg_str(
                    "tools", "read,bash,edit,write,grep,find,ls"
                ).split(",")
                if item.strip()
            ),
            timeout_seconds=self._cfg_float("timeout_seconds", 600, 10, 3600),
        )
        self._data_dir = data_dir
        self._store = SessionStateStore(
            data_dir / "session_state.json",
            settings.provider,
            settings.model,
            settings.thinking,
        )
        self._runner = PiRunner(settings)
        self._runtime_version = ""
        self._runtime_error = ""
        self._ready = False

    async def initialize(self) -> None:
        """初始化持久状态并检查 Pi 运行时。"""
        await self._store.initialize()
        if self._store.last_warning:
            logger.warning(f"[Pi控制] {self._store.last_warning}")
        try:
            self._runtime_version = await self._runner.check_runtime()
            logger.info(f"[Pi控制] Pi {self._runtime_version} 已就绪")
        except PiError as exc:
            self._runtime_error = str(exc)
            logger.error(f"[Pi控制] 运行时检查失败：{exc}")
        self._ready = True

    async def terminate(self) -> None:
        """插件卸载时停止全部活动的 Pi 子进程。"""
        self._ready = False
        await self._runner.stop_all()

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("pi")
    async def cmd_pi(self, event: AstrMessageEvent, prompt: GreedyStr = ""):
        """向当前会话的 Pi 发送任务；若已在运行则转入当前任务。"""
        self._disable_default_llm(event)
        images = await self._extract_images(event)
        if not str(prompt).strip() and not images:
            yield event.plain_result(self._help_text())
            return
        if message := self._unavailable_message():
            yield event.plain_result(message)
            return

        if await self._runner.is_active(event.unified_msg_origin):
            try:
                await self._runner.steer(
                    event.unified_msg_origin,
                    str(prompt),
                    images,
                )
            except PiNotRunningError:
                pass
            except PiError as exc:
                logger.error(f"[Pi控制] 转向失败：{exc}")
                yield event.plain_result(f"Pi 转向失败：{exc}")
                return
            else:
                yield event.plain_result("已将新指令转入当前 Pi 任务。")
                return

        record = await self._store.get_or_create(event.unified_msg_origin)
        yield event.plain_result(
            f"Pi 已接收任务，模型：{record.provider}/{record.model}，思考：{record.thinking}。"
        )
        progress_count = 0

        async def report_progress(item: ToolProgress) -> None:
            nonlocal progress_count
            if not self._cfg_bool("show_tool_progress", True) or progress_count >= 8:
                return
            try:
                if not item.finished:
                    progress_count += 1
                    await event.send(
                        MessageChain().message(self._progress_text(item, "正在使用"))
                    )
                elif item.failed:
                    progress_count += 1
                    await event.send(
                        MessageChain().message(
                            self._progress_text(item, "工具执行失败")
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - 进度推送不能中断主任务
                logger.warning(f"[Pi控制] 工具进度推送失败：{exc}")

        try:
            result = await self._runner.run_prompt(
                event.unified_msg_origin,
                record,
                str(prompt),
                report_progress,
                images,
            )
        except PiAbortedError:
            yield event.plain_result("Pi 任务已停止。")
            return
        except PiTimeoutError as exc:
            yield event.plain_result(str(exc))
            return
        except PiBusyError as exc:
            yield event.plain_result(str(exc))
            return
        except PiError as exc:
            logger.error(f"[Pi控制] 任务失败：{exc}")
            yield event.plain_result(f"Pi 执行失败：{exc}")
            return

        chunks = split_text(
            result.text, self._cfg_int("max_message_chars", 3500, 500, 10000)
        )
        for index, chunk in enumerate(chunks, start=1):
            prefix = f"【Pi {index}/{len(chunks)}】\n" if len(chunks) > 1 else ""
            yield event.plain_result(prefix + chunk)

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("pi新建")
    async def cmd_new(self, event: AstrMessageEvent):
        """为当前聊天创建新的 Pi 会话。"""
        self._disable_default_llm(event)
        if await self._runner.is_active(event.unified_msg_origin):
            return event.plain_result("当前 Pi 任务仍在运行，请先发送 /pi停止。")
        record = await self._store.rotate(event.unified_msg_origin)
        return event.plain_result(
            f"已创建新的 Pi 会话：{record.session_id[:8]}，模型保持为 "
            f"{record.provider}/{record.model}。"
        )

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("pi停止")
    async def cmd_stop(self, event: AstrMessageEvent):
        """停止当前聊天正在运行的 Pi 任务。"""
        self._disable_default_llm(event)
        try:
            await self._runner.stop(event.unified_msg_origin)
        except PiNotRunningError as exc:
            return event.plain_result(str(exc))
        return event.plain_result("已请求停止当前 Pi 任务。")

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("pi状态")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看当前聊天的 Pi 会话和用量。"""
        self._disable_default_llm(event)
        if message := self._unavailable_message():
            return event.plain_result(message)
        record = await self._store.get_or_create(event.unified_msg_origin)
        active = await self._runner.is_active(event.unified_msg_origin)
        lines = [
            f"Pi 版本：{self._runtime_version}",
            f"运行状态：{'正在执行' if active else '空闲'}",
            f"会话：{record.session_id[:8]}",
            f"模型：{record.provider}/{record.model}",
            f"思考等级：{record.thinking}",
        ]
        if not active:
            try:
                stats = await self._runner.get_stats(event.unified_msg_origin, record)
                tokens = stats.get("tokens", {})
                lines.extend(
                    [
                        f"用户消息：{stats.get('userMessages', 0)}",
                        f"工具调用：{stats.get('toolCalls', 0)}",
                        f"累计 Token：{tokens.get('total', 0)}",
                    ]
                )
                usage = stats.get("contextUsage")
                if isinstance(usage, dict) and usage.get("percent") is not None:
                    lines.append(
                        "上下文："
                        f"{usage.get('percent')}%（{usage.get('tokens')}/"
                        f"{usage.get('contextWindow')}）"
                    )
            except PiError as exc:
                lines.append(f"用量读取失败：{exc}")
        return event.plain_result("\n".join(lines))

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("pi模型")
    async def cmd_model(self, event: AstrMessageEvent, target: GreedyStr = ""):
        """列出或切换当前聊天使用的 Pi 模型。"""
        self._disable_default_llm(event)
        if message := self._unavailable_message():
            return event.plain_result(message)
        if await self._runner.is_active(event.unified_msg_origin):
            return event.plain_result("当前 Pi 任务仍在运行，暂时不能切换模型。")
        record = await self._store.get_or_create(event.unified_msg_origin)
        try:
            models = await self._runner.get_models(event.unified_msg_origin, record)
        except PiError as exc:
            return event.plain_result(f"读取 Pi 模型失败：{exc}")

        requested = str(target).strip()
        if not requested:
            rows = [
                f"- {item.get('provider')}/{item.get('id')}"
                for item in models[:30]
                if item.get("provider") and item.get("id")
            ]
            return event.plain_result(
                "当前模型："
                f"{record.provider}/{record.model}\n可用模型：\n"
                + ("\n".join(rows) if rows else "没有可用模型")
                + "\n切换：/pi模型 <供应商/模型>"
            )

        if "/" in requested:
            provider, model = requested.split("/", 1)
        else:
            provider, model = record.provider, requested
        matched = any(
            item.get("provider") == provider and item.get("id") == model
            for item in models
        )
        if not matched:
            return event.plain_result(f"未找到模型：{provider}/{model}")
        await self._store.set_model(event.unified_msg_origin, provider, model)
        return event.plain_result(f"已切换 Pi 模型：{provider}/{model}")

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("pi思考")
    async def cmd_thinking(self, event: AstrMessageEvent, level: str = ""):
        """查看或修改当前聊天的思考等级。"""
        self._disable_default_llm(event)
        record = await self._store.get_or_create(event.unified_msg_origin)
        available = list(VALID_THINKING)
        if not self._unavailable_message() and not await self._runner.is_active(
            event.unified_msg_origin
        ):
            try:
                fetched = await self._runner.get_thinking_levels(
                    event.unified_msg_origin,
                    record,
                )
                if fetched:
                    available = fetched
            except PiError as exc:
                logger.warning(f"[Pi控制] 读取思考等级失败：{exc}")
        if not level.strip():
            return event.plain_result(
                f"当前思考等级：{record.thinking}\n可选：{'、'.join(available)}"
            )
        normalized = THINKING_LEVELS.get(level.strip().lower()) or THINKING_LEVELS.get(
            level.strip()
        )
        if normalized is None:
            return event.plain_result(f"无效等级，可选：{'、'.join(available)}")
        if available and normalized not in available:
            return event.plain_result(
                f"当前模型不支持 {normalized}，可选：{'、'.join(available)}"
            )
        await self._store.set_thinking(event.unified_msg_origin, normalized)
        applied = False
        try:
            applied = await self._runner.set_thinking_active(
                event.unified_msg_origin,
                normalized,
            )
        except PiError as exc:
            logger.warning(f"[Pi控制] 运行中更新思考等级失败：{exc}")
        suffix = "，已应用到当前任务" if applied else ""
        return event.plain_result(f"已将 Pi 思考等级设为：{normalized}{suffix}")

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("pi压缩")
    async def cmd_compact(self, event: AstrMessageEvent, instruction: GreedyStr = ""):
        """压缩当前 Pi 会话上下文。"""
        self._disable_default_llm(event)
        if message := self._unavailable_message():
            return event.plain_result(message)
        if await self._runner.is_active(event.unified_msg_origin):
            return event.plain_result("当前 Pi 任务仍在运行，请先发送 /pi停止。")
        record = await self._store.get_or_create(event.unified_msg_origin)
        try:
            result = await self._runner.compact(
                event.unified_msg_origin,
                record,
                str(instruction).strip() or None,
            )
        except PiError as exc:
            return event.plain_result(f"Pi 压缩失败：{exc}")
        summary = str(result.get("summary") or "压缩完成")
        before = result.get("tokensBefore")
        after = result.get("estimatedTokensAfter")
        extra = ""
        if before is not None and after is not None:
            extra = f"\nToken：{before} → {after}"
        return event.plain_result(f"Pi 上下文已压缩。{extra}\n{summary}".strip())

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("pi帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        """显示 Pi 控制指令。"""
        self._disable_default_llm(event)
        return event.plain_result(self._help_text())

    def _unavailable_message(self) -> str:
        if not self._ready:
            return "Pi 插件尚未完成初始化，请稍后重试。"
        if self._runtime_error:
            return f"Pi 运行时不可用：{self._runtime_error}"
        return ""

    @staticmethod
    def _disable_default_llm(event: AstrMessageEvent) -> None:
        setter = getattr(event, "should_call_llm", None)
        if callable(setter):
            setter(False)

    @staticmethod
    def _progress_text(item: ToolProgress, action: str) -> str:
        detail = f"（{item.detail}）" if item.detail else ""
        return f"Pi {action}工具：{item.name}{detail}"

    def _help_text(self) -> str:
        return (
            "Pi 控制指令\n"
            "/pi <任务>：让 Pi 执行任务；任务进行中再次发送会转入当前任务\n"
            "/pi状态：查看会话、模型和用量\n"
            "/pi停止：停止当前任务\n"
            "/pi新建：开启全新会话\n"
            "/pi模型：列出可用模型\n"
            "/pi模型 <供应商/模型>：切换模型\n"
            "/pi思考 <等级>：调整思考强度\n"
            "/pi压缩：压缩当前会话上下文\n"
            "/pi帮助：显示本帮助"
        )

    async def _extract_images(self, event: AstrMessageEvent) -> list[dict[str, str]]:
        getter = getattr(event, "get_messages", None)
        if not callable(getter):
            return []
        try:
            components = getter()
        except Exception as exc:  # noqa: BLE001 - 图片提取失败时仍发送文本任务
            logger.warning(f"[Pi控制] 读取消息链失败：{exc}")
            return []
        images: list[dict[str, str]] = []
        seen: set[str] = set()
        for component in components or []:
            convert = getattr(component, "convert_to_base64", None)
            if not callable(convert):
                continue
            try:
                raw = convert()
                if inspect.isawaitable(raw):
                    raw = await raw
                item = normalize_rpc_image(str(raw or ""))
            except Exception as exc:  # noqa: BLE001 - 单张图片失败不影响其余图片
                logger.warning(f"[Pi控制] 图片转换失败：{exc}")
                continue
            if item is None or item["data"] in seen:
                continue
            seen.add(item["data"])
            images.append(item)
        return images

    def _cfg_str(self, key: str, default: str) -> str:
        value = self.config.get(key, default)
        return str(value).strip() or default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "是"}

    def _cfg_int(self, key: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(self.config.get(key, default))
        except (TypeError, ValueError):
            value = default
        return min(maximum, max(minimum, value))

    def _cfg_float(
        self,
        key: str,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        try:
            value = float(self.config.get(key, default))
        except (TypeError, ValueError):
            value = default
        return min(maximum, max(minimum, value))
