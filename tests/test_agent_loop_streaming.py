"""AgentLoop 流式模式的离线单测（change: web-sse-streaming）。

钉住五件事：
1. ``stream=True`` 时最终回复按分片以 ``[DELTA]`` 产出，最后仍有完整 ``[REPLY]``；
2. 工具轮只产出 ``[TOOL]``（名字 + 参数，参数由 Channel 层过滤），不把工具轮当回复；
3. ``stream=False`` 时 token 序列与改动前一致（只有 ``[REPLY]``），飞书和子 Agent 不受影响；
4. 护栏：首个分片前的失败会重试；首个分片后的失败不重试、直接兜底；
5. 关闭 run 生成器会一路关到模型流（客户端断开时模型停止生成）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from harness.runtime import AgentLoop
from harness.runtime.agent_loop import _FALLBACK_REPLY, RunOutcome
from harness.tools.base import Tool
from harness.tools.registry import ToolRegistry
from tests._streaming_helpers import StreamingScriptedModel, text_chunks, tool_call_chunks


async def _noop_sleep(_: float) -> None:
    return None


class _EchoArgs(BaseModel):
    value: str = Field(default="")


def _registry() -> ToolRegistry:
    async def _echo(args: _EchoArgs) -> dict:
        return {"echo": args.value}

    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="回显", args_schema=_EchoArgs, handler=_echo))
    return reg


async def _collect(loop: AgentLoop, **kwargs: Any) -> tuple[list[str], list[RunOutcome]]:
    outcomes: list[RunOutcome] = []
    tokens = [t async for t in loop.run("你好", on_outcome=outcomes.append, **kwargs)]
    return tokens, outcomes


async def test_final_reply_is_yielded_chunk_by_chunk():
    llm = StreamingScriptedModel(script=[text_chunks("你好", "，", "张三")])
    loop = AgentLoop(llm=llm, registry=ToolRegistry())

    tokens, outcomes = await _collect(loop, stream=True)

    assert tokens == ["[DELTA]你好", "[DELTA]，", "[DELTA]张三", "[REPLY]你好，张三"]
    assert outcomes == [RunOutcome.COMPLETED]


async def test_tool_round_yields_tool_token_not_deltas():
    llm = StreamingScriptedModel(script=[
        tool_call_chunks("echo", {"value": "x"}),
        text_chunks("查到了"),
    ])
    loop = AgentLoop(llm=llm, registry=_registry())

    tokens, outcomes = await _collect(loop, stream=True)

    assert tokens[0].startswith("[TOOL]")
    assert json.loads(tokens[0][len("[TOOL]"):]) == {"name": "echo", "args": {"value": "x"}}
    assert tokens[1:] == ["[DELTA]查到了", "[REPLY]查到了"]
    assert outcomes == [RunOutcome.COMPLETED]
    # 分片累加后的工具调用照常执行，结果作为工具消息传给第二轮
    tool_msgs = [m for m in llm.seen[1] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and "x" in tool_msgs[0].content


async def test_text_before_tool_call_is_streamed_but_not_the_reply():
    """模型先说一句再调工具：那句话已经推出去了，但最终回复只算最后一轮。"""
    llm = StreamingScriptedModel(script=[
        text_chunks("我查一下") + tool_call_chunks("echo", {"value": "y"}),
        text_chunks("好了"),
    ])
    loop = AgentLoop(llm=llm, registry=_registry())

    tokens, _ = await _collect(loop, stream=True)

    assert tokens[0] == "[DELTA]我查一下"
    assert tokens[1].startswith("[TOOL]")
    assert tokens[-1] == "[REPLY]好了"


async def test_non_stream_mode_is_unchanged():
    """stream 缺省 False：走 ainvoke，只有一条 [REPLY]，没有 [DELTA] / [TOOL]。"""
    llm = StreamingScriptedModel(script=[
        tool_call_chunks("echo", {"value": "x"}),
        text_chunks("查到", "了"),
    ])
    loop = AgentLoop(llm=llm, registry=_registry())

    tokens, outcomes = await _collect(loop)

    assert tokens == ["[REPLY]查到了"]
    assert outcomes == [RunOutcome.COMPLETED]
    assert llm.produced == 0  # 没走 _astream


async def test_failure_before_first_chunk_is_retried():
    llm = StreamingScriptedModel(script=[
        [ConnectionError("连接被重置")],
        text_chunks("好的"),
    ])
    loop = AgentLoop(llm=llm, registry=ToolRegistry(), retry_sleep=_noop_sleep)

    tokens, outcomes = await _collect(loop, stream=True)

    assert tokens == ["[DELTA]好的", "[REPLY]好的"]
    assert outcomes == [RunOutcome.COMPLETED]
    assert llm.calls == 2


async def test_failure_after_first_chunk_is_not_retried():
    """前半句已经推给用户，重开会让同一段话出现两遍，所以不重试、直接兜底。"""
    llm = StreamingScriptedModel(script=[
        text_chunks("前半句") + [ConnectionError("流断了")],
        text_chunks("不该被调用"),
    ])
    loop = AgentLoop(llm=llm, registry=ToolRegistry(), retry_sleep=_noop_sleep)

    tokens, outcomes = await _collect(loop, stream=True)

    assert tokens == ["[DELTA]前半句", f"[REPLY]{_FALLBACK_REPLY}"]
    assert outcomes == [RunOutcome.GUARDRAIL_EXHAUSTED]
    assert llm.calls == 1


async def test_first_chunk_timeout_exhausts_guardrail():
    llm = StreamingScriptedModel(script=[[5.0] + text_chunks("太晚了")])
    loop = AgentLoop(
        llm=llm, registry=ToolRegistry(), llm_timeout=0.05, retry_sleep=_noop_sleep
    )

    tokens, outcomes = await _collect(loop, stream=True)

    assert tokens == [f"[REPLY]{_FALLBACK_REPLY}"]
    assert outcomes == [RunOutcome.GUARDRAIL_EXHAUSTED]
    assert llm.calls == loop.llm_max_attempts


async def test_closing_run_closes_model_stream():
    """消费方中途 aclose（客户端断开）：模型流当场被关，不会把剩下的分片吐完。"""
    llm = StreamingScriptedModel(script=[text_chunks(*[f"字{i}" for i in range(50)])])
    loop = AgentLoop(llm=llm, registry=ToolRegistry())

    run = loop.run("你好", stream=True)
    got = [await run.__anext__(), await run.__anext__()]
    await run.aclose()

    assert got == ["[DELTA]字0", "[DELTA]字1"]
    assert llm.closed == 1
    assert llm.produced < 50
