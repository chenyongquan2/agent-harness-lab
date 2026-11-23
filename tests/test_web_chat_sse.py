"""/chat/stream 逐 token SSE 的端到端测试（change: web-sse-streaming）。

打真实 FastAPI 路由，LLM 换成按分片吐字的离线 fake，持久层换成内存实现。钉住：
1. 响应是 SSE：文字分片逐个成为 ``data: {"delta": ...}``，最后 ``event: done`` 带完整回复；
   响应头有 ``X-Session-Id`` 和 ``X-Accel-Buffering: no``；
2. 工具轮只推 ``event: tool``（工具名 + 提示文案），工具参数不出现在响应里；
3. 仍经 executor 的 ``execute_inline``：同会话忙时拿到排队满的兜底回复，说明同会话串行锁仍然生效；
4. 客户端断开后停止生成：模型流被关闭，会话历史补上兜底 assistant 回合。
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

import api.chat_handler as ch
from executor import TaskExecutor
from executor.local import BUSY_REPLY
from harness.memory.long_term import LongTermMemory
from harness.runtime import AgentLoop
from harness.runtime.session import SessionStore
from harness.tools.base import Tool
from harness.tools.registry import ToolRegistry
from tests._streaming_helpers import StreamingScriptedModel, text_chunks, tool_call_chunks
from tests.test_web_chat_e2e import _NoopSummaryMemory, parse_sse
from web.routes import ChatRequest, chat_stream_endpoint
from web.routes import router as web_router


class _DelegateArgs(BaseModel):
    subagent: str = Field(default="")
    task: str = Field(default="")


def _registry() -> ToolRegistry:
    """只放一个假的 delegate：名字和参数结构与生产一致，handler 不碰任何服务。"""
    async def _delegate(args: _DelegateArgs) -> str:
        return "张三明天 14:00 有空"

    reg = ToolRegistry()
    reg.register(Tool(name="delegate", description="委派", args_schema=_DelegateArgs,
                      handler=_delegate))
    return reg


@pytest.fixture
def offline(monkeypatch):
    """把 chat_handler 的模块级单例换成离线实现；返回一个「装脚本」的函数。"""
    monkeypatch.setattr(ch, "_session_store", SessionStore(repo=None))
    monkeypatch.setattr(ch, "_long_term", LongTermMemory(None))
    monkeypatch.setattr(ch, "_summary", _NoopSummaryMemory())

    def install(*script: list) -> StreamingScriptedModel:
        llm = StreamingScriptedModel(script=list(script))
        monkeypatch.setattr(ch, "_agent_loop", AgentLoop(llm=llm, registry=_registry()))
        return llm

    return install


@pytest.fixture
def client(offline):
    app = FastAPI()
    app.include_router(web_router)
    with TestClient(app) as c:
        yield c


def history(session_id: str) -> list[tuple[str, str]]:
    return [(t.role, t.content) for t in ch._session_store.get_or_create(session_id).history]


# --------------------------------------------------------------------------- #
# ① 协议与响应头
# --------------------------------------------------------------------------- #
def test_reply_is_pushed_chunk_by_chunk(client, offline):
    offline(text_chunks("好的", "，", "已为您记下。"))

    resp = client.post("/chat/stream", json={"message": "你好", "session_id": "s-sse"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["X-Session-Id"] == "s-sse"
    assert resp.headers["X-Accel-Buffering"] == "no"
    assert parse_sse(resp.text) == [
        ("message", {"delta": "好的"}),
        ("message", {"delta": "，"}),
        ("message", {"delta": "已为您记下。"}),
        ("done", {"reply": "好的，已为您记下。"}),
    ]
    # 历史只记完整回复一条，和非流式时一样
    assert history("s-sse") == [("user", "你好"), ("assistant", "好的，已为您记下。")]


def test_each_event_is_terminated_by_blank_line(client, offline):
    """前端按空行切分事件，所以每个事件都必须以空行结尾。"""
    offline(text_chunks("一", "二"))

    resp = client.post("/chat/stream", json={"message": "你好"})

    assert resp.text.endswith("\n\n")
    assert resp.text.count("\n\n") == 3  # 两段文字 + done


# --------------------------------------------------------------------------- #
# ② 工具轮：只推提示，不推参数
# --------------------------------------------------------------------------- #
def test_tool_round_pushes_label_without_args(client, offline):
    offline(
        tool_call_chunks("delegate", {"subagent": "appointment",
                                      "task": "查张三明天的档期，客户电话 13800001111"}),
        text_chunks("张三明天", "14:00 有空。"),
    )

    resp = client.post("/chat/stream", json={"message": "张三明天有空吗", "session_id": "s-tool"})

    events = parse_sse(resp.text)
    assert events[0] == ("tool", {"name": "delegate", "label": "正在查档期、处理预约…"})
    assert events[1:] == [
        ("message", {"delta": "张三明天"}),
        ("message", {"delta": "14:00 有空。"}),
        ("done", {"reply": "张三明天14:00 有空。"}),
    ]
    assert "13800001111" not in resp.text
    assert "subagent" not in resp.text


def test_unknown_tool_gets_generic_label(client, offline):
    offline(
        tool_call_chunks("delegate", {"subagent": "没有这个专员", "task": "x"}),
        text_chunks("好"),
    )

    resp = client.post("/chat/stream", json={"message": "你好"})

    assert parse_sse(resp.text)[0] == ("tool", {"name": "delegate", "label": "正在处理…"})


# --------------------------------------------------------------------------- #
# ③ 兼容端点不变、仍受 executor 的并发限制
# --------------------------------------------------------------------------- #
def test_compat_endpoint_stays_plain_text(client, offline):
    """/chat 不传 stream：走 ainvoke，没有 [DELTA] / [TOOL]，逐字节保持旧协议。"""
    llm = offline(
        tool_call_chunks("delegate", {"subagent": "appointment", "task": "x"}),
        text_chunks("好的", "。"),
    )

    resp = client.post("/chat", json={"message": "你好", "session_id": "s-compat"})

    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "[REPLY]好的。"
    assert llm.produced == 0  # 一个分片都没走流式


async def test_stream_still_goes_through_executor_accounting(offline, monkeypatch):
    """同会话已有任务在跑、排队深度为 0 时，SSE 请求拿到排队满的兜底回复。

    能拿到这句话，说明 /chat/stream 仍经过 execute_inline 的同会话串行锁。
    """
    offline(text_chunks("不该走到这里"))
    ex = TaskExecutor(ch._runner, max_queue_per_session=0)
    monkeypatch.setattr(ch, "executor", ex)

    async with ex._slot("s-busy"):  # 模拟同会话上一条消息还在处理
        resp = await chat_stream_endpoint(
            ChatRequest(message="再问一句", session_id="s-busy"), _Request()
        )
        body = "".join([chunk async for chunk in resp.body_iterator])

    assert parse_sse(body) == [("done", {"reply": BUSY_REPLY})]


# --------------------------------------------------------------------------- #
# ④ 客户端断开即停止生成
# --------------------------------------------------------------------------- #
class _Request:
    """替身 Request：第 ``disconnect_after`` 次查询起报告「客户端已断开」。"""

    def __init__(self, disconnect_after: int | None = None) -> None:
        self._after = disconnect_after
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self._after is not None and self.checks >= self._after


async def test_disconnect_stops_generation_and_keeps_history_paired(offline):
    llm = offline(text_chunks(*[f"字{i}" for i in range(50)]))
    request = _Request(disconnect_after=3)

    resp = await chat_stream_endpoint(
        ChatRequest(message="讲个长故事", session_id="s-gone"), request
    )
    body = "".join([chunk async for chunk in resp.body_iterator])

    events = parse_sse(body)
    assert events == [("message", {"delta": "字0"}), ("message", {"delta": "字1"})]
    assert llm.closed == 1          # 模型流被当场关闭
    assert llm.produced < 50        # 没把剩下的分片吐完
    # 编排层补了兜底 assistant 回合，历史仍然成对
    assert [role for role, _ in history("s-gone")] == ["user", "assistant"]


async def test_cancellation_while_waiting_stops_generation(offline):
    """长时间没有事件可推时（模型停住不动），服务器取消请求协程，同样一路关到模型流。"""
    llm = offline(text_chunks("开头") + [3600.0] + text_chunks("永远到不了"))
    resp = await chat_stream_endpoint(
        ChatRequest(message="你好", session_id="s-cancel"), _Request()
    )

    first_event = asyncio.Event()

    async def consume() -> None:
        async for _ in resp.body_iterator:
            first_event.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_event.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert llm.closed == 1
    assert [role for role, _ in history("s-cancel")] == ["user", "assistant"]
