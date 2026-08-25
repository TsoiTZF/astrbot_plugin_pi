"""供测试使用的最小 Pi RPC 协议进程。"""

from __future__ import annotations

import json
import sys

HANG_STATS = False


def emit(payload: dict, crlf: bool = False) -> None:
    ending = b"\r\n" if crlf else b"\n"
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(encoded + ending)
    sys.stdout.buffer.flush()


def response(request: dict, data: dict | None = None) -> dict:
    return {
        "id": request.get("id"),
        "type": "response",
        "command": request.get("type"),
        "success": True,
        "data": data or {},
    }


def handle_prompt(request: dict) -> None:
    global HANG_STATS
    message = str(request.get("message", ""))
    images = request.get("images") or []
    if message == "损坏协议":
        sys.stdout.buffer.write(b"{not-json}\n")
        sys.stdout.buffer.flush()
        return

    emit(response(request), crlf=True)
    emit({"type": "agent_start"})
    if message in {"等待停止", "等待超时"}:
        wait_for_control()
        return
    if message == "忽略停止":
        drain_until_killed()
        return
    if message == "没有文本":
        emit({"type": "agent_settled"})
        return
    if message == "卡住统计":
        HANG_STATS = True
        finish_text("统计已完成")
        return
    if message == "扩展确认":
        emit(
            {
                "type": "extension_ui_request",
                "id": "ui-1",
                "method": "confirm",
                "title": "允许执行？",
                "message": "危险命令",
            }
        )
        if not wait_for_ui_cancel():
            return

    emit(
        {
            "type": "tool_execution_start",
            "toolCallId": "tool-1",
            "toolName": "read",
            "args": {"path": "README.md"},
        }
    )
    emit(
        {
            "type": "tool_execution_end",
            "toolCallId": "tool-1",
            "toolName": "read",
            "result": {"content": [{"type": "text", "text": "完成"}]},
            "isError": False,
        }
    )
    emit(
        {
            "type": "message_update",
            "usage": {"input": 1, "output": 1},
            "assistantMessageEvent": {
                "type": "text_delta",
                "contentIndex": 0,
                "delta": "流式旧文本",
            },
        }
    )
    final_text = f"测试完成：{message}"
    if images:
        mime = str(images[0].get("mimeType") or "")
        final_text += f"：{mime}"
    finish_text(final_text)


def finish_text(text: str) -> None:
    emit(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        }
    )
    emit({"type": "agent_settled"})


def wait_for_control() -> None:
    for raw in sys.stdin.buffer:
        request = json.loads(raw.decode("utf-8"))
        command = request.get("type")
        if command == "abort":
            emit(response(request))
            emit({"type": "agent_settled"})
            return
        if command == "steer":
            emit(response(request))
            finish_text(f"已转向：{request.get('message', '')}")
            return
        if command == "set_thinking_level":
            emit(response(request))
            continue
        if command == "extension_ui_response":
            continue


def wait_for_ui_cancel() -> bool:
    for raw in sys.stdin.buffer:
        request = json.loads(raw.decode("utf-8"))
        command = request.get("type")
        if command == "extension_ui_response" and request.get("cancelled"):
            return True
        if command == "abort":
            emit(response(request))
            emit({"type": "agent_settled"})
            return False
    return False


def drain_until_killed() -> None:
    for _raw in sys.stdin.buffer:
        pass


def main() -> None:
    if "--version" in sys.argv:
        print("0.84.2-fake")
        return
    for raw in sys.stdin.buffer:
        request = json.loads(raw.decode("utf-8"))
        command = request.get("type")
        if command == "prompt":
            handle_prompt(request)
        elif command == "abort":
            emit(response(request))
            emit({"type": "agent_settled"})
        elif command == "get_available_models":
            emit(
                response(
                    request,
                    {
                        "models": [
                            {
                                "provider": "91grok",
                                "id": "grok-4.5",
                                "reasoning": True,
                            },
                            {
                                "provider": "openai",
                                "id": "gpt-test",
                                "reasoning": False,
                            },
                        ]
                    },
                )
            )
        elif command == "get_available_thinking_levels":
            emit(response(request, {"levels": ["low", "medium", "high"]}))
        elif command == "get_session_stats":
            if HANG_STATS:
                sys.stdin.buffer.read()
                return
            emit(
                response(
                    request,
                    {
                        "sessionId": "fake-session",
                        "userMessages": 2,
                        "assistantMessages": 2,
                        "toolCalls": 1,
                        "tokens": {"total": 42},
                        "contextUsage": {
                            "tokens": 60,
                            "contextWindow": 200,
                            "percent": 30,
                        },
                    },
                )
            )
        elif command == "compact":
            emit(
                response(
                    request,
                    {
                        "summary": "已压缩历史上下文",
                        "tokensBefore": 100,
                        "estimatedTokensAfter": 20,
                    },
                )
            )
        else:
            emit(
                {
                    "id": request.get("id"),
                    "type": "response",
                    "command": command,
                    "success": False,
                    "error": "不支持的测试命令",
                }
            )


if __name__ == "__main__":
    main()
