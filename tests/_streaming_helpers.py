"""流式测试共用的离线 fake（change: web-sse-streaming）。

``StreamingScriptedModel`` 每次被调用时取脚本里的下一段，按分片逐个吐出：
- ``AIMessageChunk``：原样作为一个分片；
- ``Exception`` 实例：在该位置抛出（模拟连接断开）；
- ``float``：在该位置 ``asyncio.sleep`` 这么多秒（模拟模型停住不动，用来测超时）。

同时实现 ``_generate``（把同一段脚本拼成整条消息），所以 ``ainvoke`` 路径也能跑，
用来证明 ``stream=False`` 时行为与改动前一致。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def text_chunks(*parts: str) -> list:
    """一段纯文字回复，按给定的几段切分。"""
    return [AIMessageChunk(content=p) for p in parts]


def tool_call_chunks(name: str, args: dict, call_id: str = "call_1") -> list:
    """一次工具调用，参数 JSON 拆成两个分片吐出（真实模型就是这样分段给参数的）。"""
    raw = json.dumps(args, ensure_ascii=False)
    half = len(raw) // 2
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": name, "args": raw[:half], "id": call_id, "index": 0}],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": None, "args": raw[half:], "id": None, "index": 0}],
        ),
    ]


class StreamingScriptedModel(BaseChatModel):
    """按脚本逐分片吐出的 fake 模型；记录调用次数、吐出的分片数、流是否被关闭。"""

    script: List[list] = []
    calls: int = 0
    produced: int = 0
    closed: int = 0
    seen: List[List[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "streaming-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StreamingScriptedModel":
        return self

    def _next_script(self, messages: List[BaseMessage]) -> list:
        idx = min(self.calls, len(self.script) - 1)
        object.__setattr__(self, "calls", self.calls + 1)
        object.__setattr__(self, "seen", self.seen + [list(messages)])
        return self.script[idx]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        acc = None
        for item in self._next_script(messages):
            if isinstance(item, Exception):
                raise item
            if isinstance(item, AIMessageChunk):
                acc = item if acc is None else acc + item
        msg = AIMessage(
            content=acc.content if acc else "",
            tool_calls=acc.tool_calls if acc else [],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        script = self._next_script(messages)
        try:
            for item in script:
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, float):
                    await asyncio.sleep(item)
                    continue
                object.__setattr__(self, "produced", self.produced + 1)
                yield ChatGenerationChunk(message=item)
                await asyncio.sleep(0)  # 让出事件循环，像真实网络流一样每个分片之间有 await 点
        finally:
            object.__setattr__(self, "closed", self.closed + 1)
