"""
Pre-LLM-call context window check — triggers compaction when conversation is too large.

For an overview of all context management mechanisms, see:
docs/reference/context-management.md
"""

import logging
import time
from typing import Any, Optional

import sentry_sdk
from pydantic import BaseModel

from holmes.common.env_vars import ENABLE_CONVERSATION_HISTORY_COMPACTION
from holmes.core.llm import (
    LLM,
    ContextWindowUsage,
    get_context_window_compaction_threshold_pct,
)
from holmes.core.llm_usage import RequestStats
from holmes.core.truncation.compaction import compact_conversation_history
from holmes.utils.stream import StreamEvents, StreamMessage


def check_compaction_needed(
    llm: "LLM", messages: list[dict], tools: Optional[list[dict[str, Any]]]
) -> Optional[StreamMessage]:
    """Check if compaction is needed and return a COMPACTION_START event if so.

    This is separated from compact_if_necessary so the caller can yield
    the START event to the SSE stream *before* the blocking compaction call.
    """
    # 如果整体关闭了“对话历史压缩”能力，就直接返回 None，
    # 调用方据此知道这轮不需要发 compaction start 事件。
    if not ENABLE_CONVERSATION_HISTORY_COMPACTION:
        return None

    # 先估算当前 messages + tools 占用的 token 数。
    # 这里提前算，是为了在真正进入压缩前就能判断是否接近上下文窗口上限。
    initial_tokens = llm.count_tokens(messages=messages, tools=tools)  # type: ignore
    # 取模型的总上下文窗口大小。
    max_context_size = llm.get_context_window_size()
    # 取模型预留的最大输出 token。
    # 这里必须把输出空间也算进去，因为输入即使勉强塞得下，若不给回答预留空间也无法正常生成结果。
    maximum_output_token = llm.get_maximum_output_token()

    # 判断当前输入 token + 预留输出 token 是否已经超过“触发压缩”的阈值。
    # 注意这里不是和 max_context_size 直接比较，而是和一个可配置百分比阈值比较，
    # 这样做是为了更早触发压缩，给后续模型回答留出安全余量，避免到了硬上限才处理。
    if (initial_tokens.total_tokens + maximum_output_token) > (
        max_context_size * get_context_window_compaction_threshold_pct() / 100
    ):
        # 顺手记录当前消息条数，便于前端展示和后续排查。
        num_messages = len(messages)
        # 返回一个“即将开始压缩”的事件，而不是在这里直接做压缩。
        # 原因是上层可以先把这个事件推给 SSE / UI，再进入可能较慢的压缩逻辑，
        # 这样用户能及时看到系统正在做什么，减少“卡住没反应”的感觉。
        return StreamMessage(
            event=StreamEvents.CONVERSATION_HISTORY_COMPACTION_START,
            data={
                "content": f"Compacting conversation history ({initial_tokens.total_tokens} tokens, {num_messages} messages)...",
                "metadata": {
                    "initial_tokens": initial_tokens.total_tokens,
                    "num_messages": num_messages,
                    "max_context_size": max_context_size,
                    "threshold_pct": get_context_window_compaction_threshold_pct(),
                },
            },
        )
    # 没超过阈值就不需要压缩，也不需要发任何事件。
    return None


class CompactionInsufficientError(Exception):
    """Raised when conversation compaction was not sufficient to fit the context window."""

    def __init__(self, message: str, events: list[StreamMessage], compaction_usage: Optional[RequestStats] = None):
        super().__init__(message)
        self.events = events
        self.compaction_usage = compaction_usage


class ContextWindowLimiterOutput(BaseModel):
    metadata: dict
    messages: list[dict]
    events: list[StreamMessage]
    max_context_size: int
    maximum_output_token: int
    tokens: ContextWindowUsage
    conversation_history_compacted: bool
    compaction_usage: Optional["RequestStats"] = None


@sentry_sdk.trace
def compact_if_necessary(
    llm: LLM, messages: list[dict], tools: Optional[list[dict[str, Any]]]
) -> ContextWindowLimiterOutput:
    # 记录函数开始时间，后面用于输出整段压缩判断/执行流程的耗时日志。
    t0 = time.monotonic()
    # 收集本函数产生的流式事件，例如“压缩完成”或“压缩失败”的提示事件。
    events = []
    # 预留给上层使用的额外元数据容器；当前函数里没有填充业务字段，但统一保留这个返回位。
    metadata = {}
    # 先计算当前 messages + tools 一共占用了多少 token。
    # 这是后续判断是否需要压缩、以及压缩是否成功的基础数据。
    initial_tokens = llm.count_tokens(messages=messages, tools=tools)  # type: ignore
    # 读取模型的最大上下文窗口大小。
    max_context_size = llm.get_context_window_size()
    # 读取模型允许的最大输出 token。
    # 之所以要单独拿出来，是因为输入不能只看“当前消息塞不塞得下”，
    # 还必须为模型最终生成回答预留空间。
    maximum_output_token = llm.get_maximum_output_token()
    # 标记本次是否真的执行并采用了对话压缩结果。
    conversation_history_compacted = False
    # 记录压缩过程本身消耗的 token / cost，用于最终统计与错误透传。
    compaction_usage = RequestStats()
    # 只有在“启用了历史压缩”且“当前输入 + 预留输出”超过阈值时，才尝试压缩。
    # 这里使用阈值百分比，而不是等到顶满上下文窗口再处理，是为了提前留出安全余量。
    if ENABLE_CONVERSATION_HISTORY_COMPACTION and (
        initial_tokens.total_tokens + maximum_output_token
    ) > (max_context_size * get_context_window_compaction_threshold_pct() / 100):
        # 记录压缩前消息条数，后面用于生成统计信息。
        num_messages_before = len(messages)
        # 真正执行对话历史压缩，通常会把较早的历史总结成更短的摘要形式。
        compaction_result = compact_conversation_history(
            original_conversation_history=messages, llm=llm
        )
        # 保存压缩动作本身的 usage，后面会写入最终输出。
        compaction_usage = compaction_result.usage
        # 对压缩后的消息重新计算 token，用于判断压缩是否真的起到了效果。
        compacted_tokens = llm.count_tokens(compaction_result.messages_after_compaction, tools=tools)
        compacted_total_tokens = compacted_tokens.total_tokens

        # 只有压缩后的 token 数真的变少了，才接受这次压缩结果。
        if compacted_total_tokens < initial_tokens.total_tokens:
            # 用压缩后的消息替换原消息，后续流程都基于这份新历史继续。
            messages = compaction_result.messages_after_compaction
            # 记录压缩后的消息条数。
            num_messages_after = len(messages)
            # 计算压缩比例，便于日志和前端展示。
            compression_ratio = round((1 - compacted_total_tokens / initial_tokens.total_tokens) * 100, 1)
            # 生成一条人类可读的压缩结果说明。
            compaction_message = f"The conversation history has been compacted from {initial_tokens.total_tokens} to {compacted_total_tokens} tokens"
            logging.info(compaction_message)
            # 标记本次确实采用了压缩结果。
            conversation_history_compacted = True

            # Extract the LLM-generated summary from the compacted messages
            # Structure is: [system_prompt?, last_user_prompt?, assistant_summary, continuation_marker]
            # 尝试从压缩后的消息里提取 LLM 生成的摘要内容。
            # 这里默认 assistant 消息里承载的是压缩摘要，方便后续展示给客户端。
            compaction_summary = None
            for msg in compaction_result.messages_after_compaction:
                if msg.get("role") == "assistant":
                    compaction_summary = msg.get("content")
                    break

            # 组装压缩统计信息，供事件 metadata 和上层 UI 使用。
            compaction_stats: dict = {
                "initial_tokens": initial_tokens.total_tokens,
                "compacted_tokens": compacted_total_tokens,
                "compression_ratio_pct": compression_ratio,
                "num_messages_before": num_messages_before,
                "num_messages_after": num_messages_after,
                "max_context_size": max_context_size,
                "threshold_pct": get_context_window_compaction_threshold_pct(),
            }
            if compaction_usage:
                # 如果压缩本身调用了模型，也把对应成本信息透出。
                compaction_stats["compaction_cost"] = {
                    "total_cost": compaction_usage.total_cost,
                    "prompt_tokens": compaction_usage.prompt_tokens,
                    "completion_tokens": compaction_usage.completion_tokens,
                    "total_tokens": compaction_usage.total_tokens,
                }

            # 发出“压缩完成”事件，供前端展示结构化压缩结果。
            events.append(
                StreamMessage(
                    event=StreamEvents.CONVERSATION_HISTORY_COMPACTED,
                    data={
                        "content": compaction_message,
                        "compaction_summary": compaction_summary,
                        "messages": compaction_result.messages_after_compaction,
                        "metadata": compaction_stats,
                    },
                )
            )
            # 再补一条普通 AI_MESSAGE，兼容只消费通用文本事件的客户端。
            events.append(
                StreamMessage(
                    event=StreamEvents.AI_MESSAGE,
                    data={"content": compaction_message},
                )
            )
        else:
            # 如果压缩后 token 不减反增，说明这次压缩没有实际价值，只记录错误日志，不采用结果。
            logging.error(
                f"Failed to reduce token count when compacting conversation history. Original tokens:{initial_tokens.total_tokens}. Compacted tokens:{compacted_total_tokens}"
            )

    # 不管前面有没有尝试压缩，最后都要基于当前 messages 重新计算一次最终 token。
    tokens = llm.count_tokens(messages=messages, tools=tools)  # type: ignore
    # 如果即使经过压缩（或未启用压缩），当前输入 + 预留输出 仍然超过上下文窗口，
    # 说明这轮请求已经无法安全发送给模型，需要明确报错并提示用户开启新会话。
    if (tokens.total_tokens + maximum_output_token) > max_context_size:
        if ENABLE_CONVERSATION_HISTORY_COMPACTION:
            # 启用了压缩却仍放不下，说明“压缩不足以解决问题”。
            failure_msg = (
                f"Conversation history compaction failed to reduce tokens sufficiently. "
                f"Current: {tokens.total_tokens} tokens + {maximum_output_token} max output = "
                f"{tokens.total_tokens + maximum_output_token}, but context window is {max_context_size}. "
                f"Please start a new conversation."
            )
        else:
            # 压缩功能关闭时，则明确告诉调用方：超窗的根因是没有启用压缩。
            failure_msg = (
                f"Conversation history exceeds the context window and compaction is disabled. "
                f"Current: {tokens.total_tokens} tokens + {maximum_output_token} max output = "
                f"{tokens.total_tokens + maximum_output_token}, but context window is {max_context_size}. "
                f"Please start a new conversation."
            )
        logging.error(failure_msg)
        # 把失败信息作为普通 AI_MESSAGE 发出去，保证客户端至少能看到人类可读的错误说明。
        events.append(
            StreamMessage(
                event=StreamEvents.AI_MESSAGE,
                data={"content": failure_msg},
            )
        )
        # 抛出专用异常，并把事件列表和压缩 usage 一并带出去，
        # 让上层既能显示错误，也能保留压缩阶段的成本统计。
        raise CompactionInsufficientError(failure_msg, events=events, compaction_usage=compaction_usage)

    # 记录整个函数总耗时与最终 token 数，便于性能排查。
    elapsed_ms = (time.monotonic() - t0) * 1000
    logging.debug(f"compact_if_necessary: {elapsed_ms:.1f}ms total | {tokens.total_tokens} tokens")

    # 返回统一结构，包含：
    # 当前可继续使用的 messages、压缩相关事件、token 信息，以及是否发生过压缩等状态。
    return ContextWindowLimiterOutput(
        events=events,
        messages=messages,
        metadata=metadata,
        max_context_size=max_context_size,
        maximum_output_token=maximum_output_token,
        tokens=tokens,
        conversation_history_compacted=conversation_history_compacted,
        compaction_usage=compaction_usage,
    )
