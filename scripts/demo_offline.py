"""离线演示：不需要 API 密钥，用一个按剧本回答的假模型驱动真实的 AgentLoop。

运行：

    uv run python scripts/demo_offline.py

演示三种情况，每种都会打印最终回复、工具调用顺序和这次请求的追踪事件：

1. 正常完成：先查空档，再预约，最后给出回复。
2. 工具出错：工具抛异常，错误被当成工具结果交还给模型，循环不会崩溃。
3. 原地重复：模型一直发出完全相同的工具调用，连续 3 次后护栏停止循环。

这里的工具和模型都是演示用的替身，只为展示运行时本身（循环、护栏、追踪）。
参考业务域里真实的工具在 domains/appointment/tools/ 下。
"""

import asyncio
import os
import sys
from typing import Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认不是 UTF-8

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from harness.observability.exporter import InMemoryExporter  # noqa: E402
from harness.observability.tracer import Tracer  # noqa: E402
from harness.runtime import AgentLoop  # noqa: E402
from harness.tools.base import Tool  # noqa: E402
from harness.tools.registry import ToolRegistry  # noqa: E402


class ScriptedModel(BaseChatModel):
    """按预设的顺序返回回复；脚本用完后一直重复最后一条。"""

    responses: List[AIMessage] = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-demo-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class SlotArgs(BaseModel):
    time: str = Field(description="预约时间，例如 15:00")


def build_registry(fail_on_check: bool = False) -> ToolRegistry:
    async def check_slot(args: SlotArgs) -> str:
        if fail_on_check:
            raise RuntimeError("排班服务暂时不可用")
        return f"{args.time} 有空档"

    async def book(args: SlotArgs) -> str:
        return f"已预约 {args.time}，预约号 A-001"

    registry = ToolRegistry()
    registry.register(Tool("check_slot", "查询某个时间是否有空档", SlotArgs, check_slot))
    registry.register(Tool("book", "预约某个时间", SlotArgs, book))
    return registry


async def run_case(title: str, responses: List[AIMessage], registry: ToolRegistry) -> None:
    exporter = InMemoryExporter()
    loop = AgentLoop(llm=ScriptedModel(responses=responses), registry=registry, tracer=Tracer(exporter))
    reply = "".join([token async for token in loop.run("帮我约明天下午 3 点")])

    print(f"\n=== {title} ===")
    print(f"最终回复：{reply}")
    print("追踪事件：")
    for span in exporter.spans:
        for event in span.events:
            detail = {k: v for k, v in event.payload.items() if k in ("name", "args", "type", "text", "result")}
            print(f"  [{span.name}] {event.kind}: {detail}")


async def main() -> None:
    await run_case(
        "1. 正常完成",
        [
            call("check_slot", {"time": "15:00"}, "c1"),
            call("book", {"time": "15:00"}, "c2"),
            AIMessage(content="已为你预约明天 15:00，预约号 A-001。"),
        ],
        build_registry(),
    )
    await run_case(
        "2. 工具出错，错误交还给模型",
        [
            call("check_slot", {"time": "15:00"}, "c1"),
            AIMessage(content="抱歉，排班服务暂时不可用，请稍后再试。"),
        ],
        build_registry(fail_on_check=True),
    )
    await run_case(
        "3. 原地重复，护栏停止循环",
        [call("check_slot", {"time": "15:00"}, "c1")],
        build_registry(),
    )


if __name__ == "__main__":
    asyncio.run(main())
