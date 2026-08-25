# AstrBot Pi 控制插件

通过 AstrBot 聊天消息控制 Pi 编码智能体。插件直接使用 Pi 官方 JSONL/RPC 协议，不解析终端界面，也不需要用户操作 CLI。

## 功能

- 每个 AstrBot 会话拥有独立的 Pi 会话和工作区。
- AstrBot 或容器重启后继续原会话。
- 支持任务停止、运行中转向、会话重建、上下文压缩、状态与 Token 查询。
- 支持列出和切换 Pi 模型、按当前模型调整思考等级。
- 消息中的图片会随任务发给 Pi。
- Pi 工具执行时可推送进度，长回复自动分段。
- 扩展弹出的确认框会自动取消，避免无界面环境卡死。
- 插件卸载或停用时会关闭全部活动子进程。

## 指令

所有指令仅限 AstrBot 管理员使用。

```text
/pi <任务>
/pi状态
/pi停止
/pi新建
/pi模型
/pi模型 <供应商/模型>
/pi思考 <关|最小|低|中|高|极高|最大>
/pi压缩 [说明]
/pi帮助
```

任务进行中再次发送 `/pi` 会把新指令转入当前任务，而不是拒绝。`/pi停止` 会先请求 Pi 中止，若进程仍不退出则强制结束。

## 运行目录

当前部署使用以下持久化路径：

```text
/AstrBot/data/pi_runtime       Pi npm 运行时
/AstrBot/data/plugin_data/astrbot_plugin_pi/pi_config
                               Pi 模型与凭据配置
/AstrBot/data/plugin_data/astrbot_plugin_pi/sessions
                               Pi 会话文件
/AstrBot/data/pi_workspaces    按聊天隔离的工作区
```

API 密钥不放在 AstrBot 插件配置中。请使用 Pi 自身的 `models.json` 或 `auth.json`，插件只引用 Pi 配置目录。

当前部署默认使用 Grok 4.5，默认思考等级为 `high`。该模型通常只开放 `low`、`medium`、`high` 三档；其他模型若支持更多档位，可通过 `/pi思考` 查看。

## 本地验证

```powershell
pytest -q
ruff check .
ruff format --check .
python -m compileall -q .
```
