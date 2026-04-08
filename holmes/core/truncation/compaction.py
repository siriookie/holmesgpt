"""
LLM-based conversation history compaction — summarizes old messages to free context space.

For an overview of all context management mechanisms, see:
docs/reference/context-management.md
"""

import logging
from typing import Any, Optional

from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from holmes.core.llm import LLM
from holmes.core.llm_usage import RequestStats
from holmes.plugins.prompts import load_and_render_prompt


class CompactionResult(BaseModel):
    """Result of conversation history compaction."""

    messages_after_compaction: list[dict]
    usage: Optional[RequestStats] = None


def strip_system_prompt(
    conversation_history: list[dict],
) -> tuple[list[dict], Optional[dict]]:
    if not conversation_history:
        return conversation_history, None
    first_message = conversation_history[0]
    if first_message and first_message.get("role") == "system":
        return conversation_history[1:], first_message
    return conversation_history[:], None


def find_last_user_prompt(conversation_history: list[dict]) -> Optional[dict]:
    if not conversation_history:
        return None
    last_user_prompt: Optional[dict] = None
    for message in conversation_history:
        if message.get("role") == "user":
            last_user_prompt = message
    return last_user_prompt


def _count_image_tokens_in_messages(messages: list[dict], llm: LLM) -> int:
    """Count total tokens used by image blocks across all messages."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        # Count tokens for a synthetic message containing only image blocks
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image_url"]
        if image_blocks:
            synthetic = {"role": "user", "content": image_blocks}
            total += llm.count_tokens(messages=[synthetic]).total_tokens
    return total


def _strip_images_for_compaction(messages: list[dict]) -> list[dict]:
    """Strip image_url blocks from messages, replacing with a count placeholder."""
    stripped: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            stripped.append(msg)
            continue
        new_content: list[dict[str, Any]] = []
        image_count = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                image_count += 1
            else:
                new_content.append(block)
        if image_count > 0:
            new_content.append({
                "type": "text",
                "text": f"[{image_count} image(s) were present but stripped from compaction]",
            })
        new_msg = dict(msg)
        new_msg["content"] = new_content
        new_msg.pop("token_count", None)
        stripped.append(new_msg)
    return stripped


def compact_conversation_history(
    original_conversation_history: list[dict], llm: LLM
) -> CompactionResult:
    """
    The compacted conversation history contains:
      1. Original system prompt, uncompacted (if present)
      2. Last user prompt, uncompacted (if present)
      3. Compacted conversation history (role=assistant)
      4. Compaction message (role=system)
    """
    # 先把 system prompt 从原始历史里拆出来，避免把系统提示词也拿去做摘要压缩。
    # 原因是 system prompt 往往是行为约束，必须原样保留，不能被总结改写。
    conversation_history, system_prompt_message = strip_system_prompt(
        original_conversation_history
    )
    # 加载“如何压缩对话历史”的提示词模板，作为后面交给 LLM 的压缩指令。
    compaction_instructions = load_and_render_prompt(
        prompt="builtin://conversation_history_compaction.jinja2", context={}
    )

    # Decide whether to keep images in the compaction input.
    # Keep them if the conversation (with images) fits in the compaction LLM's
    # context window, so it can describe what was in them. Otherwise strip them.
    # Include instruction tokens in the budget since they are appended before the LLM call.
    # 取压缩所用模型的上下文窗口大小。
    context_window = llm.get_context_window_size()
    # 取模型最大输出 token，保证给压缩摘要本身预留生成空间。
    maximum_output_token = llm.get_maximum_output_token()
    # 单独计算压缩指令本身会占多少 token，因为它稍后会作为一条 user 消息附加到历史后面。
    instruction_tokens = llm.count_tokens(
        messages=[{"role": "user", "content": compaction_instructions}]
    ).total_tokens
    # 计算当前待压缩对话历史的总 token。
    total_tokens = llm.count_tokens(messages=conversation_history).total_tokens
    # 单独估算图片内容占用的 token。
    # 这样后面可以决定：是保留图片让模型一起总结，还是为了省上下文把图片剥离掉。
    image_tokens = _count_image_tokens_in_messages(conversation_history, llm)

    # 如果存在图片，并且“历史 + 压缩指令 + 预留输出”仍然能放进上下文窗口，
    # 就保留图片，让模型有机会在摘要里提到图片信息。
    if image_tokens > 0 and (total_tokens + instruction_tokens + maximum_output_token) <= context_window:
        logging.info(
            f"Compaction: keeping {image_tokens} image tokens "
            f"(conversation fits in context window: {total_tokens} + {instruction_tokens} + {maximum_output_token} <= {context_window})"
        )
    elif image_tokens > 0:
        # 如果图片会导致压缩请求本身超窗，就先把图片去掉。
        # 这样做的核心目的是优先保证“压缩能成功执行”，哪怕损失一部分多模态细节。
        logging.info(
            f"Compaction: stripping {image_tokens} image tokens "
            f"(conversation would overflow: {total_tokens} + {instruction_tokens} + {maximum_output_token} > {context_window})"
        )
        conversation_history = _strip_images_for_compaction(conversation_history)

    # 把压缩指令追加成最后一条 user 消息，交给模型对整段历史做总结。
    conversation_history.append({"role": "user", "content": compaction_instructions})

    # 调用模型生成压缩后的摘要。
    response: ModelResponse = llm.completion(
        messages=conversation_history, drop_params=True
    )  # type: ignore
    # 提取这次压缩调用本身的 usage / cost。
    compaction_usage = RequestStats.from_response(response)

    # 先初始化响应消息占位，下面会从 LLM 返回结果里提取真正的 assistant 摘要消息。
    response_message = None
    if (
        response
        and response.choices
        and response.choices[0]
        and response.choices[0].message  # type:ignore
    ):
        # 正常情况下，压缩结果就在第一条 choice 的 message 中。
        response_message = response.choices[0].message  # type:ignore
    else:
        # 如果压缩响应结构异常，就记录错误并回退到原始历史，
        # 避免返回一份损坏或不完整的压缩结果。
        logging.error(
            "Failed to compact conversation history. Unexpected LLM's response for compaction"
        )
        return CompactionResult(messages_after_compaction=original_conversation_history, usage=compaction_usage)

    # 开始组装“压缩后的新会话历史”。
    compacted_conversation_history: list[dict] = []
    if system_prompt_message:
        # 把原始 system prompt 放回最前面，确保系统约束继续生效。
        compacted_conversation_history.append(system_prompt_message)

    # 找出原始历史里的最后一条 user prompt，并原样保留。
    # 这样做是为了让模型在压缩后仍然能直接看到用户当前最新的问题，而不是只看到摘要。
    last_user_prompt = find_last_user_prompt(original_conversation_history)
    if last_user_prompt:
        compacted_conversation_history.append(last_user_prompt)

    # 把模型生成的压缩摘要作为 assistant 消息加入新历史。
    compacted_conversation_history.append(
        response_message.model_dump(
            exclude_defaults=True, exclude_unset=True, exclude_none=True
        )
    )

    # 追加一条系统消息，明确告诉后续模型：
    # 之前的历史已经被压缩，现在应基于摘要继续对话。
    compacted_conversation_history.append(
        {
            "role": "system",
            "content": "The conversation history has been compacted to preserve available space in the context window. Continue.",
        }
    )
    # 返回压缩后的新历史，以及压缩动作本身的 usage 统计。
    return CompactionResult(
        messages_after_compaction=compacted_conversation_history, usage=compaction_usage
    )
