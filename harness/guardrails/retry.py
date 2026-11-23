"""LLM 调用护栏：超时 + 指数退避重试（Phase 5）。

``guarded_invoke`` 对一次异步 LLM 调用施加：
- **超时**：单次调用经 ``asyncio.wait_for`` 限时，超时即视为本次失败；
- **重试**：对超时与瞬时连接类异常按指数退避（``base_delay × 2**n``）重试，最多
  ``max_attempts`` 次；退避等待经可注入的 ``sleep`` 完成（测试传 no-op，不真睡）。

重试耗尽后抛出明确的 :class:`GuardrailExhausted`（而非让底层异常冒泡），由调用方
（``AgentLoop``）捕获并转为优雅降级的兜底回复。

设计要点（见 OpenSpec change phase-5-guardrails 的 design.md D1/D2）：超时与重试逻辑
紧耦合，合在本文件内（timeout 作为 retry 的参数）；只包裹**只读、幂等**的 LLM 调用，
带副作用的工具调用 MUST NOT 经此重试（避免重复执行写操作）。

【核心哲学：不对称重试】LLM 调用可重试、工具调用绝不重试。原因：LLM 调用没有副作用
（重发一次顶多多花点钱、多等一会），值得在瞬时故障时重试；而工具可能写库/下单/发消息，
重发会「重复执行」造成真实损害，故工具失败只做错误隔离、绝不重试。本文件只服务前者；
后者的处理见 ``AgentLoop._dispatch``。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Optional, TypeVar

# T = call 的返回类型；用 TypeVar 让 guarded_invoke「传进去什么类型、返回什么类型」，
# 调用方拿到的不是 Any 而是精确类型（如 AIMessage），IDE/类型检查器能正确推断。
T = TypeVar("T")

# 默认护栏参数（``AgentLoop`` 可经构造参数覆盖）。
# 抽成模块级常量而非写死在签名里：AgentLoop 也 import 它们做默认值，单一真相源。
DEFAULT_TIMEOUT = 30.0        # 单次 LLM 调用最多等 30 秒
DEFAULT_MAX_ATTEMPTS = 3      # 含首次，最多试 3 次
DEFAULT_BASE_DELAY = 0.5      # 退避基准：0.5 → 1 → 2s（见下方 base_delay × 2**attempt）

# 可重试的瞬时异常：超时与连接类。其它异常视为不可重试，直接冒泡。
# 为什么只这几类：它们是「重发一次很可能就好」的临时性故障；像参数错、鉴权失败
# 这类「重发一万次也一样」的异常不在此列——把它们也重试只是白白浪费时间和钱。
# 易误解点：列表里同时有 asyncio.TimeoutError 和 TimeoutError——在不同 Python 版本/
# 场景下超时可能以两种类型之一抛出，都列上以防漏网。
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
)


class GuardrailExhausted(Exception):
    """LLM 调用经超时 + 重试护栏后仍失败，重试已耗尽。"""
    # 自定义异常类型而非复用底层异常：让上层（AgentLoop）能用一句
    # ``except GuardrailExhausted`` 精确捕获「护栏判定的彻底失败」，
    # 与代码里其它偶发异常区分开，进而走「优雅兜底回复」而不是崩掉请求。


async def guarded_invoke(
    call: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
) -> T:
    """带超时与指数退避重试地执行一次异步调用。

    Args:
        call: 零参 async thunk，每次重试都重新调用它发起一次新的请求。
        timeout: 单次调用的超时秒数。
        max_attempts: 最大尝试次数（含首次）；≥1。
        base_delay: 指数退避基准秒数；第 n 次重试前等待 ``base_delay × 2**(n-1)``。
        sleep: 退避等待实现（默认 ``asyncio.sleep``）；测试可注入 no-op 避免真实等待。

    Returns:
        ``call`` 成功时的返回值。

    Raises:
        GuardrailExhausted: 所有尝试均因可重试异常失败时抛出，``__cause__`` 为最后一次异常。
    """
    # sleep 可注入是关键设计：生产用真的 asyncio.sleep（真等），测试传一个「立即返回」
    # 的 no-op，于是单测能验证「退避逻辑被走到」却不必真的 sleep 0.5/1/2 秒拖慢测试。
    sleep_fn = sleep or asyncio.sleep
    # 记住最后一次异常，耗尽时塞进 GuardrailExhausted 的 __cause__，方便排障溯源。
    last_exc: Optional[BaseException] = None

    # ════════════════════════════════════════════════════════════════════════
    # 重试主循环：用 range 而非 while——尝试次数天然带硬上限，绝不会无限重试
    # attempt 从 0 数起：0=首次, 1=第一次重试, ...，共 max_attempts 次
    # ════════════════════════════════════════════════════════════════════════
    for attempt in range(max_attempts):
        try:
            # 每次都重新调 call() 发起一次全新请求（call 是 thunk，故可重复触发）；
            # wait_for 给本次调用套上超时——超时即抛 TimeoutError，落入下面的 except。
            return await asyncio.wait_for(call(), timeout=timeout)
        except RETRYABLE_EXCEPTIONS as exc:
            # 只有「可重试」的瞬时异常才会被这里接住；其它异常不在元组内，
            # 不会被 except 命中 → 直接冒泡出去（符合「不可重试就别重试」的设计）。
            last_exc = exc
            # 还有剩余尝试则退避后重试；否则跳出转 GuardrailExhausted。
            # 易误解点：最后一次失败「后」不再 sleep（再睡也没有下一次了），故用
            # ``attempt + 1 < max_attempts`` 守卫——只在「还有下一次」时才等待。
            if attempt + 1 < max_attempts:
                # 指数退避：base_delay × 2**attempt → 0.5, 1, 2, ...
                # 为什么越等越久：给下游/网络留出恢复时间，避免「失败就猛刷」加剧拥塞。
                await sleep_fn(base_delay * (2 ** attempt))

    # 走到这里 = for 循环跑满 max_attempts 次仍每次都落进 except（从未 return）。
    # 抛 GuardrailExhausted 而非让 last_exc 冒泡：给上层一个「语义明确、可统一捕获」
    # 的失败信号。``from last_exc`` 保留原始异常链（traceback 里能看到根因）。
    raise GuardrailExhausted(
        f"LLM 调用重试 {max_attempts} 次仍失败：{last_exc!r}"
    ) from last_exc


async def guarded_stream(
    open_stream: Callable[[], AsyncIterator[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
) -> AsyncGenerator[T, None]:
    """``guarded_invoke`` 的流式版本：给一次流式 LLM 调用加超时与重试（change: web-sse-streaming）。

    与 ``guarded_invoke`` 的两处不同，都由「分片已经推给用户」这件事决定：

    - **超时按分片计**：等首个分片最多 ``timeout`` 秒，之后相邻两个分片之间也最多等
      ``timeout`` 秒。不对整次调用限时——长回复本来就要写几十秒，按整体限时会把正常的
      长回复掐断。
    - **只在首个分片之前重试**：首个分片到达之前失败，用户什么都没看到，重开一次流和
      ``guarded_invoke`` 重发请求等价；首个分片之后再失败，前半段文字已经推到前端，重开
      会让同一段话出现两遍，所以直接抛 :class:`GuardrailExhausted`，由 ``AgentLoop`` 给
      兜底回复。

    Args:
        open_stream: 零参函数，每次调用开一条新的流（如 ``lambda: llm.astream(messages)``）。
        其余参数同 ``guarded_invoke``。

    Yields:
        流里的分片，原样传出。

    Raises:
        GuardrailExhausted: 首个分片之前重试耗尽，或首个分片之后遇到可重试类异常。
    """
    sleep_fn = sleep or asyncio.sleep
    last_exc: Optional[BaseException] = None

    # ① 等首个分片：这一段失败可以安全重试（用户还没看到任何文字）。
    for attempt in range(max_attempts):
        stream = open_stream()
        try:
            first = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return  # 空流：模型什么都没返回，交给调用方当空回复处理
        except RETRYABLE_EXCEPTIONS as exc:
            await _close_quietly(stream)
            last_exc = exc
            if attempt + 1 < max_attempts:
                await sleep_fn(base_delay * (2 ** attempt))
            continue
        except BaseException:
            await _close_quietly(stream)
            raise
        break
    else:
        raise GuardrailExhausted(
            f"LLM 流式调用重试 {max_attempts} 次仍未拿到首个分片：{last_exc!r}"
        ) from last_exc

    # ② 首个分片已到：逐个原样传出，后续失败不再重试。
    #    finally 里关流：调用方提前退出（客户端断开）时，这里关掉底层 HTTP 流，模型侧随之停止生成。
    try:
        yield first
        while True:
            try:
                chunk = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except RETRYABLE_EXCEPTIONS as exc:
                raise GuardrailExhausted(
                    f"LLM 流式输出在首个分片之后中断，已推出的文字无法撤回，不再重试：{exc!r}"
                ) from exc
            yield chunk
    finally:
        await _close_quietly(stream)


async def _close_quietly(stream: AsyncIterator[Any]) -> None:
    """关闭一条流；流本身没有 ``aclose``（普通异步迭代器）时跳过。"""
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()
