import concurrent.futures
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Named logger for user-facing display messages (tool progress, AI messages, etc.)
# In interactive mode this logger is silenced; the CLI renders from stream events instead.
display_logger = logging.getLogger("holmes.display.tool_calling_llm")

import sentry_sdk
from openai import BadRequestError
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel, Field

from holmes.common.env_vars import (
    LOG_LLM_USAGE_RESPONSE,
    RESET_REPEATED_TOOL_CALL_CHECK_AFTER_COMPACTION,
    TEMPERATURE,
    load_bool,
)
from holmes.core.llm import LLM
from holmes.core.llm_usage import RequestStats

from holmes.core.models import (
    FrontendToolResult,
    PendingFrontendToolCall,
    PendingToolApproval,
    ToolApprovalDecision,
    ToolCallResult,
)
from holmes.core.safeguards import prevent_overly_repeated_tool_call
from holmes.core.tools import (
    StructuredToolResult,
    StructuredToolResultStatus,
    ToolInvokeContext,
)
from holmes.core.tools_utils.tool_context_window_limiter import (
    spill_oversized_tool_result,
)
from holmes.core.tools_utils.tool_executor import ToolExecutor
from holmes.core.tracing import DummySpan
from holmes.core.truncation.input_context_window_limiter import (
    CompactionInsufficientError,
    check_compaction_needed,
    compact_if_necessary,
)
from holmes.utils.colors import AI_COLOR
from holmes.utils.stream import (
    StreamEvents,
    StreamMessage,
    add_token_count_to_metadata,
    build_stream_event_token_count,
)
from holmes.utils.tags import parse_messages_tags

class LLMInterruptedError(Exception):
    """Raised when the user interrupts an in-progress LLM call (e.g. via Escape key)."""

    pass


# Create a named logger for cost tracking
cost_logger = logging.getLogger("holmes.costs")


def _extract_text_from_content(content: Any) -> str:
    """Extract text from message content, handling both string and array formats.

    OpenAI/LiteLLM message content can be:
    - A plain string: "some text"
    - An array of content objects: [{"type": "text", "text": "some text", ...}]

    The array format is used by our cache_control feature (see llm.py add_cache_control_to_last_message)
    which converts string content to a single-item array. For tool messages, there's always
    only one text item containing the full tool output with tool_call_metadata at the start.

    Args:
        content: Message content (string or array)

    Returns:
        Extracted text as a string
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Tool messages have a single text item (created by format_tool_result_data,
        # possibly wrapped in array by cache_control). Return the first text item.
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")

    return ""


def extract_bash_session_prefixes(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract bash session approved prefixes from conversation history.

    Scans tool result messages for bash_session_approved_prefixes stored in
    tool_call_metadata. These prefixes were approved by the user via the
    "Yes, and don't ask again" option.

    Args:
        messages: Conversation history messages

    Returns:
        List of approved prefixes accumulated from all tool results
    """
    prefixes: set[str] = set()

    for msg in messages:
        if msg.get("role") != "tool":
            continue

        content = _extract_text_from_content(msg.get("content", ""))
        if not content:
            continue

        # Extract tool_call_metadata from the content string
        # Format: tool_call_metadata={"tool_name": "...", ...}
        match = re.search(r"tool_call_metadata=(\{[^}]+\})", content)
        if not match:
            continue

        try:
            metadata = json.loads(match.group(1))
            if "bash_session_approved_prefixes" in metadata:
                prefixes.update(metadata["bash_session_approved_prefixes"])
        except (json.JSONDecodeError, KeyError):
            continue

    if prefixes:
        logging.info(
            f"Found {len(prefixes)} session-approved bash prefixes from conversation: {list(prefixes)}"
        )
    return list(prefixes)


# Callback type: receives a pending approval, returns (approved, optional_feedback)
ApprovalCallback = Callable[[PendingToolApproval], tuple[bool, Optional[str]]]


class LLMResult(RequestStats):
    tool_calls: Optional[List[ToolCallResult]] = None
    num_llm_calls: Optional[int] = None   # 一共调了几轮 LLM
    result: Optional[str] = None           # 最终答案文本
    unprocessed_result: Optional[str] = None
    instructions: List[str] = Field(default_factory=list)
    messages: Optional[List[dict]] = None  # 完整对话历史（含工具调用记录）
    metadata: Optional[Dict[Any, Any]] = None


class ToolCallWithDecision(BaseModel):
    message_index: int
    tool_call: ChatCompletionMessageToolCall
    decision: Optional[ToolApprovalDecision]


class ToolCallingLLM:
    llm: LLM

    def __init__(
        self,
        tool_executor: ToolExecutor,
        max_steps: int,
        llm: LLM,
        tool_results_dir: Optional[Path],
        tracer=None,
    ):
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.tracer = tracer
        self.llm = llm
        self.tool_results_dir = tool_results_dir

        self._runbook_in_use: bool = False

    def with_executor(self, tool_executor: ToolExecutor) -> "ToolCallingLLM":
        """Return a shallow copy with a different ToolExecutor.

        Used to inject per-request frontend tools via a cloned executor
        without mutating the shared ToolCallingLLM instance.
        """
        clone = ToolCallingLLM(
            tool_executor=tool_executor,
            max_steps=self.max_steps,
            llm=self.llm,
            tool_results_dir=self.tool_results_dir,
            tracer=self.tracer,
        )
        # Preserve transient state so resumed turns keep access to
        # runbook-unlocked (restricted) tools.
        clone._runbook_in_use = self._runbook_in_use
        return clone

    def reset_interaction_state(self) -> None:
        """
        For interactive loop, reset runbooks in use
        """
        self._runbook_in_use = False

    def _supports_vision(self) -> bool:
        """Check if vision/multimodal input is enabled.

        Always True unless explicitly disabled via HOLMES_DISABLE_VISION=true.
        """
        return not load_bool("HOLMES_DISABLE_VISION", False)

    def _has_bash_for_file_access(self) -> bool:
        """Check if bash toolset is available for reading saved tool result files."""
        for toolset in self.tool_executor.enabled_toolsets:
            if toolset.name == "bash":
                config = toolset.config
                if config:
                    return config.builtin_allowlist != "none"
                return False
        return False

    def _execute_tool_decisions(
        self,
        messages: List[Dict[str, Any]],
        tool_decisions: List[ToolApprovalDecision],
        request_context: Optional[Dict[str, Any]] = None,
        trace_span: Any = None,
    ) -> tuple[List[Dict[str, Any]], list[StreamMessage]]:
        """Execute approved tools and record rejections for denied ones.

        Called after the user (CLI callback or HTTP client) has decided on each
        pending tool call. Re-invokes approved tools with user_approved=True,
        and injects denial errors for rejected ones.

        Returns:
            Updated messages list with tool execution results and stream events.
        """
        # 如果上层没有传 trace span，就退化成空 span，保证后续工具执行埋点逻辑可统一复用。
        if trace_span is None:
            trace_span = DummySpan()

        # 收集本次“审批后恢复执行”阶段要回放给客户端的事件。
        events: list[StreamMessage] = []
        # 没有任何审批结果时，直接原样返回，不做额外处理。
        if not tool_decisions:
            return messages, events

        # Create decision lookup
        # 把用户对每个 tool_call 的审批结果按 tool_call_id 建索引，
        # 这样后面在消息历史里回找 pending tool call 时可以 O(1) 取到对应决定。
        decisions_by_tool_call_id = {
            decision.tool_call_id: decision for decision in tool_decisions
        }

        # 收集消息历史里所有仍处于 pending_approval 状态的工具调用，
        # 后面会逐个根据 decision 执行或构造拒绝结果。
        pending_tool_calls: list[ToolCallWithDecision] = []

        # 倒序扫描 messages。
        # 这样做的原因是：后面会往 assistant tool_call 请求后面插入 tool result，
        # 倒序处理可以减少前面插入对后续 message_index 的影响。
        for i in reversed(range(len(messages))):
            msg = messages[i]
            # 只关心带 tool_calls 的 assistant 消息，因为 pending approval 标记挂在这里。
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                message_tool_calls = msg.get("tool_calls", [])
                for tool_call in message_tool_calls:
                    # 先按 tool_call id 查找用户的审批结果；理论上每个 pending tool_call 都应有对应 decision。
                    decision = decisions_by_tool_call_id.get(tool_call.get("id"), None)
                    if tool_call.get("pending_approval"):
                        # 一旦准备恢复处理，就先把 pending_approval 标记清掉，
                        # 避免这条消息在未来再次被当成“还没审批”的请求重复处理。
                        del tool_call[
                            "pending_approval"
                        ]  # Cleanup so that a pending approval is not tagged on message in a future response
                        # 保存恢复执行所需的完整信息：
                        # 原始 tool_call、用户 decision、以及这条 assistant 消息所在位置。
                        pending_tool_calls.append(
                            ToolCallWithDecision(
                                tool_call=ChatCompletionMessageToolCall(**tool_call),
                                decision=decision,
                                message_index=i,
                            )
                        )

        if not pending_tool_calls:
            # 如果客户端传回了 decision，但历史里已经找不到 pending approval，
            # 说明消息状态和客户端状态不同步了，这属于逻辑异常，直接报错更安全。
            error_message = f"Received {len(tool_decisions)} tool decisions but no pending approvals found"
            logging.error(error_message)
            raise Exception(error_message)
        # Extract existing session prefixes from conversation history
        # 恢复会话级 bash 已批准前缀。
        # 这样“本次批准并记住前缀”的效果可以立刻影响同一次恢复流程中的后续命令。
        session_prefixes = extract_bash_session_prefixes(messages)

        # 逐个处理待恢复的工具调用：批准则真正执行，拒绝则构造一个错误结果返回给模型。
        for tool_call_with_decision in pending_tool_calls:
            # 取出原始工具调用对象。
            tool_call = tool_call_with_decision.tool_call
            # 取出这个工具调用对应的审批决定。
            decision = tool_call_with_decision.decision
            # 先初始化结果变量，后面无论批准还是拒绝都要产出一个 ToolCallResult。
            tool_result: Optional[ToolCallResult] = None
            if decision and decision.approved:
                # 用户批准后，重新真正执行这条工具调用。
                # 这里显式传 user_approved=True，避免再次卡在审批环节。
                tool_result = self._invoke_llm_tool_call(
                    tool_to_call=tool_call,
                    previous_tool_calls=[],
                    trace_span=trace_span,
                    tool_number=None,
                    user_approved=True,
                    session_approved_prefixes=session_prefixes,
                    request_context=request_context,
                    enable_tool_approval=True,  # always True when processing decisions
                )
            else:
                # Tool was rejected or no decision found, add rejection message
                # 用户拒绝，或者根本没有找到 decision 时，统一转成一个错误型 tool result。
                # 这样模型后续能读到“这条工具调用被用户拒绝”这个事实，而不是静默消失。
                feedback_text = f" User feedback: {decision.feedback}" if decision and decision.feedback else ""
                tool_result = ToolCallResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    description=tool_call.function.name,
                    result=StructuredToolResult(
                        status=StructuredToolResultStatus.ERROR,
                        error=f"Tool execution was denied by the user.{feedback_text}",
                    ),
                )

            # 不论最终是批准执行还是拒绝，都向客户端发一个 TOOL_RESULT 事件。
            # 这样前端/CLI 才能完整展示恢复阶段发生了什么。
            events.append(
                StreamMessage(
                    event=StreamEvents.TOOL_RESULT,
                    data=tool_result.to_client_dict(),
                )
            )

            # If user chose "Yes, and don't ask again", include prefixes in metadata
            # 如果用户选择了“批准并记住前缀”，就把这些前缀写进 tool result metadata，
            # 后续 extract_bash_session_prefixes 可以从消息历史中把它们恢复出来。
            extra_metadata = None
            if decision and decision.approved and decision.save_prefixes:
                logging.info(
                    f"Saving bash session prefixes for future commands: {decision.save_prefixes}"
                )
                extra_metadata = {
                    "bash_session_approved_prefixes": decision.save_prefixes
                }

            # 把工具结果转成 LLM 可消费的 tool message，准备插回对话历史。
            tool_call_message = tool_result.to_llm_message(
                extra_metadata=extra_metadata,
                supports_vision=self._supports_vision(),
            )

            # It is expected that the tool call result directly follows the tool call request from the LLM
            # The API call may contain a user ask which is appended to the messages so we can't just append
            # tool call results; they need to be inserted right after the llm's message requesting tool calls
            # 这里必须 insert，而不是 append。
            # 原因是 OpenAI 风格的消息序列要求：tool result 必须紧跟在发起该 tool_call 的 assistant 消息之后。
            # 如果简单 append，而这期间消息末尾又有新的 user 消息，消息顺序就会错，模型可能无法正确关联工具结果。
            messages.insert(
                tool_call_with_decision.message_index + 1, tool_call_message
            )

        # 返回插入了恢复后 tool result 的最新消息历史，以及需要回放给客户端的事件列表。
        return messages, events

    @staticmethod
    def _process_frontend_tool_results(
        messages: List[Dict[str, Any]],
        frontend_tool_results: List[FrontendToolResult],
    ) -> tuple[List[Dict[str, Any]], list[StreamMessage]]:
        """Inject frontend tool results into the conversation history.

        Called when the client sends results for tools it executed locally.
        Finds the pending frontend tool calls in messages, clears their
        pending flag, and inserts tool result messages.

        Returns:
            Updated messages list and stream events for each result.
        """
        events: list[StreamMessage] = []
        if not frontend_tool_results:
            return messages, events

        results_by_id = {r.tool_call_id: r for r in frontend_tool_results}
        matched_ids: set[str] = set()

        # Find pending frontend tool calls in messages (reverse to insert correctly)
        for i in reversed(range(len(messages))):
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool_call in msg.get("tool_calls", []):
                    if not tool_call.get("pending_frontend"):
                        continue

                    tool_call_id = tool_call.get("id")
                    result = results_by_id.get(tool_call_id)
                    if tool_call_id:
                        matched_ids.add(tool_call_id)
                    if not result:
                        logging.warning(
                            f"No frontend result for pending tool call {tool_call.get('id')}"
                        )
                        # Insert an error so the LLM knows
                        tool_result_msg = {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": tool_call.get("function", {}).get("name", "unknown"),
                            "content": "Error: frontend did not return a result for this tool call.",
                        }
                    else:
                        tool_result_msg = {
                            "tool_call_id": result.tool_call_id,
                            "role": "tool",
                            "name": result.tool_name,
                            "content": result.result,
                        }

                    # Clean up the pending flag
                    del tool_call["pending_frontend"]

                    # Insert result right after the assistant message
                    messages.insert(i + 1, tool_result_msg)

                    tool_result = ToolCallResult(
                        tool_call_id=tool_call["id"],
                        tool_name=tool_call.get("function", {}).get("name", "unknown"),
                        description=f"Frontend tool: {tool_call.get('function', {}).get('name', 'unknown')}",
                        result=StructuredToolResult(
                            status=StructuredToolResultStatus.SUCCESS if result else StructuredToolResultStatus.ERROR,
                            data=result.result if result else None,
                            error="Frontend did not return a result" if not result else None,
                        ),
                    )
                    events.append(
                        StreamMessage(
                            event=StreamEvents.TOOL_RESULT,
                            data=tool_result.to_client_dict(),
                        )
                    )

        # Warn about results that didn't match any pending frontend tool call
        unmatched = set(results_by_id.keys()) - matched_ids
        if unmatched:
            logging.warning(
                f"Frontend tool results provided for unknown tool_call_ids (ignored): {unmatched}"
            )

        return messages, events

    def _should_include_restricted_tools(self) -> bool:
        """Check if restricted tools should be included in the tools list."""
        return self._runbook_in_use

    def _get_tools(self) -> list:
        """Get tools list, filtering restricted tools based on authorization."""
        return self.tool_executor.get_all_tools_openai_format(
            include_restricted=self._should_include_restricted_tools(),
        )

    @sentry_sdk.trace
    def call(  # type: ignore
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        trace_span=DummySpan(),
        tool_number_offset: int = 0,
        request_context: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[threading.Event] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> LLMResult:
        """Synchronous wrapper around call_stream(). Drains the generator
        and reconstructs an LLMResult."""

        all_tool_calls: list[dict] = []
        tool_decisions: Optional[List[ToolApprovalDecision]] = None
        total_num_llm_calls = 0
        accumulated_stats = RequestStats()

        while True:
            stream = self.call_stream(
                msgs=messages,
                response_format=response_format,
                enable_tool_approval=approval_callback is not None,
                tool_decisions=tool_decisions,
                trace_span=trace_span,
                cancel_event=cancel_event,
                tool_number_offset=tool_number_offset,
                request_context=request_context,
                iteration_offset=total_num_llm_calls,
            )

            tool_decisions = None
            terminal_data = None
            terminal_event = None
            start_tool_count = 0
            saw_tool_results = False

            for event in stream:
                # Log blank line when a tool batch ends (transition away from TOOL_RESULT)
                if saw_tool_results and event.event != StreamEvents.TOOL_RESULT:
                    display_logger.info("")
                    saw_tool_results = False

                if event.event == StreamEvents.START_TOOL:
                    start_tool_count += 1
                elif event.event == StreamEvents.TOOL_RESULT:
                    tool_number_offset += 1
                    saw_tool_results = True
                    all_tool_calls.append(event.data)
                    if start_tool_count > 0:
                        display_logger.info(
                            f"The AI requested [bold]{start_tool_count}[/bold] tool call(s)."
                        )
                        start_tool_count = 0
                elif event.event == StreamEvents.AI_MESSAGE:
                    reasoning = event.data.get("reasoning")
                    content = event.data.get("content")
                    if reasoning:
                        display_logger.info(
                            f"[italic dim]AI reasoning:\n\n{reasoning}[/italic dim]\n"
                        )
                    if content and content.strip():
                        display_logger.info(
                            f"[bold {AI_COLOR}]AI:[/bold {AI_COLOR}] {content}"
                        )
                elif event.event in (StreamEvents.ANSWER_END, StreamEvents.APPROVAL_REQUIRED):
                    terminal_data = event.data
                    terminal_event = event.event
                    break

            if not terminal_data:
                raise Exception("Stream ended without ANSWER_END or APPROVAL_REQUIRED")

            # call_stream returns the absolute iteration count (including offset),
            # so we assign rather than accumulate to avoid double-counting.
            total_num_llm_calls = terminal_data.get("num_llm_calls", 0)
            accumulated_stats += RequestStats(**terminal_data.get("costs", {}))

            if terminal_event == StreamEvents.APPROVAL_REQUIRED:
                # Check if there are frontend tool calls — can't execute in sync mode
                pending_frontend = terminal_data.get("pending_frontend_tool_calls", [])
                if pending_frontend:
                    logging.warning(
                        "Frontend tool calls requested but no frontend available in sync mode. "
                        f"Pending: {[fc['tool_name'] for fc in pending_frontend]}"
                    )
                    return LLMResult(
                        result="Investigation paused: the AI requested frontend-defined tools that cannot be executed in sync mode.",
                        tool_calls=all_tool_calls,  # type: ignore
                        num_llm_calls=total_num_llm_calls,
                        messages=terminal_data.get("messages"),
                        metadata=terminal_data.get("metadata"),
                        **accumulated_stats.model_dump(),
                    )

                # Only approval pauses — prompt via callback and continue
                messages = terminal_data["messages"]
                tool_decisions = self._prompt_for_approval_decisions(
                    terminal_data["pending_approvals"],
                    approval_callback,
                )
                continue

            # ANSWER_END — deduplicate tool calls keeping last per ID
            deduped: dict[str, dict] = {}
            for tc in all_tool_calls:
                deduped[tc.get("tool_call_id", id(tc))] = tc
            return LLMResult(
                result=terminal_data["content"],
                tool_calls=list(deduped.values()),
                num_llm_calls=total_num_llm_calls,
                messages=terminal_data["messages"],
                metadata=terminal_data.get("metadata"),
                **accumulated_stats.model_dump(),
            )

    def _prompt_for_approval_decisions(
        self,
        pending_approvals: List[dict],
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> List[ToolApprovalDecision]:
        """Prompt the user for approval decisions on each pending tool call.

        For CLI: the approval_callback shows an interactive menu per tool.
        When a user approves one tool with "save prefix", a subsequent tool
        in the same batch with the same prefix is auto-approved (re-check).
        """
        decisions: List[ToolApprovalDecision] = []
        for approval_dict in pending_approvals:
            approval = PendingToolApproval(**approval_dict)

            # Re-check: a previous approval in this batch may have saved
            # the prefix to disk, making this tool no longer need approval.
            if self._is_tool_call_already_approved(approval.tool_name, approval.params):
                logging.debug(f"Approval no longer needed for {approval.tool_name}")
                decisions.append(ToolApprovalDecision(
                    tool_call_id=approval.tool_call_id,
                    approved=True,
                ))
                continue

            if not approval_callback:
                decisions.append(ToolApprovalDecision(
                    tool_call_id=approval.tool_call_id,
                    approved=False,
                ))
                continue

            approved, feedback = approval_callback(approval)
            decisions.append(ToolApprovalDecision(
                tool_call_id=approval.tool_call_id,
                approved=approved,
                feedback=feedback if not approved else None,
            ))

        return decisions

    def _directly_invoke_tool_call(
        self,
        tool_name: str,
        tool_params: dict,
        user_approved: bool,
        tool_call_id: str,
        tool_number: Optional[int] = None,
        session_approved_prefixes: Optional[List[str]] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> StructuredToolResult:
        # Ensure the toolset is initialized (lazy initialization on first use)
        init_error = self.tool_executor.ensure_toolset_initialized(tool_name)
        if isinstance(init_error, str):
            return StructuredToolResult(
                status=StructuredToolResultStatus.ERROR,
                error=init_error,
                params=tool_params,
            )

        tool = self.tool_executor.get_tool_by_name(tool_name)
        if not tool:
            logging.warning(
                f"Skipping tool execution for {tool_name}: args: {tool_params}"
            )
            return StructuredToolResult(
                status=StructuredToolResultStatus.ERROR,
                error=f"Failed to find tool {tool_name}",
                params=tool_params,
            )

        try:
            invoke_context = ToolInvokeContext(
                tool_number=tool_number,
                user_approved=user_approved,
                llm=self.llm,
                max_token_count=self.llm.get_max_token_count_for_single_tool(),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                session_approved_prefixes=session_approved_prefixes or [],
                request_context=request_context,
            )
            tool_response = tool.invoke(tool_params, context=invoke_context)

            # Track runbook usage - if fetch_runbook is called successfully,
            # restricted tools become available for the rest of the current request
            if (
                tool_name == "fetch_runbook"
                and tool_response.status == StructuredToolResultStatus.SUCCESS
            ):
                self._runbook_in_use = True
                logging.debug("Runbook fetched - restricted tools now available")

        except Exception as e:
            logging.error(
                f"Tool call to {tool_name} failed with an Exception", exc_info=True
            )
            tool_response = StructuredToolResult(
                status=StructuredToolResultStatus.ERROR,
                error=f"Tool call failed: {e}",
                params=tool_params,
            )
        return tool_response

    @staticmethod
    def _log_tool_call_result(
        tool_span,
        tool_call_result: ToolCallResult,
        approval_possible=True,
        original_token_count=None,
        image_count=0,
    ):
        tool_span.set_attributes(name=tool_call_result.tool_name)
        status = tool_call_result.result.status

        if (
            status == StructuredToolResultStatus.APPROVAL_REQUIRED
            and not approval_possible
        ):
            status = StructuredToolResultStatus.ERROR

        if status == StructuredToolResultStatus.ERROR:
            error = (
                tool_call_result.result.error
                if tool_call_result.result.error
                else "Unspecified error"
            )
        else:
            error = None

        # Include images in output if present (before spill clears them)
        images = tool_call_result.result.images
        if images:
            output = {
                "data": tool_call_result.result.data,
                "images": [{"mimeType": img.get("mimeType", ""), "data_length": len(img.get("data", ""))} for img in images],
            }
        else:
            output = tool_call_result.result.data

        metadata = {
            "status": status,
            "description": tool_call_result.description,
            "return_code": tool_call_result.result.return_code,
            "error": tool_call_result.result.error,
            "original_token_count": original_token_count,
        }
        if image_count > 0:
            metadata["image_count"] = image_count

        tool_span.log(
            input=tool_call_result.result.params,
            output=output,
            error=error,
            metadata=metadata,
        )

    def _invoke_llm_tool_call(
        self,
        tool_to_call: ChatCompletionMessageToolCall,
        previous_tool_calls: list[dict],
        trace_span=None,
        tool_number=None,
        user_approved: bool = False,
        session_approved_prefixes: Optional[List[str]] = None,
        request_context: Optional[Dict[str, Any]] = None,
        enable_tool_approval: bool = False,
    ) -> ToolCallResult:
        if trace_span is None:
            trace_span = DummySpan()
        with trace_span.start_span(type="tool") as tool_span:
            # ChatCompletionMessageToolCall is a union of FunctionToolCall (has 'function')
            # and CustomToolCall (has 'custom'). We only support function tool calls.
            if not hasattr(tool_to_call, "function"):
                logging.error(f"Unsupported custom tool call: {tool_to_call}")
                tool_call_result = ToolCallResult(
                    tool_call_id=tool_to_call.id,
                    tool_name="Unknown_Custom_Tool",
                    description="NA",
                    result=StructuredToolResult(
                        status=StructuredToolResultStatus.ERROR,
                        error="Custom tool calls are not supported",
                        params=None,
                    ),
                )
                ToolCallingLLM._log_tool_call_result(tool_span, tool_call_result, enable_tool_approval)
                return tool_call_result

            tool_name = tool_to_call.function.name
            tool_arguments = tool_to_call.function.arguments
            tool_id = tool_to_call.id

            tool_params = {}
            try:
                tool_params = json.loads(tool_arguments)
            except Exception:
                logging.warning(
                    f"Failed to parse arguments for tool: {tool_name}. args: {tool_arguments}"
                )

            tool_response = None
            if not user_approved:
                tool_response = prevent_overly_repeated_tool_call(
                    tool_name=tool_name,
                    tool_params=tool_params,
                    tool_calls=previous_tool_calls,
                )

            if not tool_response:
                tool_response = self._directly_invoke_tool_call(
                    tool_name=tool_name,
                    tool_params=tool_params,
                    user_approved=user_approved,
                    tool_number=tool_number,
                    tool_call_id=tool_id,
                    session_approved_prefixes=session_approved_prefixes,
                    request_context=request_context,
                )

            tool = self.tool_executor.get_tool_by_name(tool_name)
            toolset_name = self.tool_executor.get_toolset_name(tool_name)
            tool_call_result = ToolCallResult(
                tool_call_id=tool_id,
                tool_name=tool_name,
                description=str(tool.get_parameterized_one_liner(tool_params))
                if tool
                else "",
                result=tool_response,
                toolset_name=toolset_name if isinstance(toolset_name, str) else None,
            )

            # Save image count before spill_oversized_tool_result clears them
            image_count = len(tool_call_result.result.images) if tool_call_result.result.images else 0

            # See docs/reference/context-management.md for how this fits with compaction
            original_token_count = spill_oversized_tool_result(
                tool_call_result=tool_call_result,
                llm=self.llm,
                tool_results_dir=self.tool_results_dir
                if self.tool_results_dir and self._has_bash_for_file_access()
                else None,
            )

            ToolCallingLLM._log_tool_call_result(
                tool_span,
                tool_call_result,
                enable_tool_approval,
                original_token_count,
                image_count,
            )
            return tool_call_result

    def _is_tool_call_already_approved(
        self,
        tool_name: str,
        params: dict,
        session_approved_prefixes: Optional[List[str]] = None,
    ) -> bool:
        """Check whether a tool call would pass approval without user interaction.

        Checks both static allow lists (config + CLI-saved prefixes) and
        optionally session-approved prefixes from the conversation history.
        """
        tool = self.tool_executor.get_tool_by_name(tool_name)
        if not tool:
            return False
        context = ToolInvokeContext(
            llm=self.llm,
            max_token_count=self.llm.get_max_token_count_for_single_tool(),
            tool_name=tool_name,
            tool_call_id="",
            session_approved_prefixes=session_approved_prefixes or [],
        )
        approval = tool.requires_approval(params, context)
        return not approval or not approval.needs_approval

    def _emit_token_count(
        self,
        messages: list[dict],
        tools: Optional[list],
        full_response: Any,
        limit_result: Any,
        metadata: Dict[Any, Any],
        stats: RequestStats,
    ) -> StreamMessage:
        """Build a TOKEN_COUNT event with current token usage and costs."""
        tokens = self.llm.count_tokens(messages=messages, tools=tools)
        add_token_count_to_metadata(
            tokens=tokens,
            full_llm_response=full_response,
            max_context_size=limit_result.max_context_size,
            maximum_output_token=limit_result.maximum_output_token,
            metadata=metadata,
        )
        metadata["costs"] = stats.model_dump()
        return build_stream_event_token_count(metadata=metadata)

    def call_stream(
        self,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        msgs: Optional[list[dict]] = None,
        enable_tool_approval: bool = False,
        tool_decisions: List[ToolApprovalDecision] | None = None,
        frontend_tool_results: Optional[List[FrontendToolResult]] = None,
        request_context: Optional[Dict[str, Any]] = None,
        trace_span: Any = None,
        cancel_event: Optional[threading.Event] = None,
        tool_number_offset: int = 0,
        iteration_offset: int = 0,
    ):
        """
        This function DOES NOT call llm.completion(stream=true).
        This function streams holmes one iteration at a time instead of waiting for all iterations to complete.

        Frontend tools: Frontend tools are registered as FrontendPauseTool instances
        in the ToolExecutor (via clone_with_extra_tools). When the LLM calls one,
        it returns FRONTEND_PAUSE status. call_stream handles this by pausing the
        stream with an APPROVAL_REQUIRED event containing pending_frontend_tool_calls.
        The client executes the tool and resumes by sending frontend_tool_results.
        """
        # 如果调用方没有传 trace span，就退化成一个空实现，保证后续埋点代码不用到处判空。
        if trace_span is None:
            trace_span = DummySpan()

        # 汇总本次流式调用全过程中产生过的所有工具调用结果。
        # 最终 ANSWER_END 会把它们一并返回给上层。
        all_tool_calls: list[dict] = []

        # Process tool decisions if provided (approval resume)
        # 如果这是一次“审批后恢复”的调用，那么先把上一轮等待用户确认的工具执行掉，
        # 并把对应结果重新注入到消息历史里，然后再继续后续 LLM 迭代。
        if msgs and tool_decisions:
            logging.info(f"Processing {len(tool_decisions)} tool decisions")
            msgs, events = self._execute_tool_decisions(
                msgs, tool_decisions, request_context, trace_span=trace_span
            )
            for ev in events:
                # 先把恢复阶段产生的事件原样流给调用方，这样前端/CLI 能看到完整过程。
                yield ev
                # Collect tool results from approval re-invocations
                if ev.event == StreamEvents.TOOL_RESULT:
                    # 这些是审批通过后真正执行出来的工具结果，也应计入最终总结果。
                    all_tool_calls.append(ev.data)

        # Process frontend tool results if provided (frontend tool resume)
        # 如果这是一次“前端工具执行后恢复”的调用，先把客户端回传的工具结果补回消息历史，
        # 这样模型下一轮推理就能基于这些结果继续往下走。
        if msgs and frontend_tool_results:
            logging.info(f"Processing {len(frontend_tool_results)} frontend tool results")
            msgs, events = self._process_frontend_tool_results(msgs, frontend_tool_results)
            for ev in events:
                # 同样要把恢复阶段的事件继续往外发，保证事件流完整。
                yield ev
                if ev.event == StreamEvents.TOOL_RESULT:
                    # 前端工具结果也要进入本轮 all_tool_calls 汇总。
                    all_tool_calls.append(ev.data)

        # 拷贝一份消息列表作为当前工作上下文，避免直接修改调用方传入的原列表引用。
        messages: list[dict] = list(msgs) if msgs else []
        # 记录当前这次推理链路中已经执行过的工具调用，用于重复调用检测等逻辑。
        tool_calls: list[dict] = []
        # 取出当前可供模型调用的工具列表。
        # 注意后续 runbook 激活后，这个列表可能会动态刷新。
        tools: Optional[list] = self._get_tools()
        # 最大允许的 LLM 迭代步数，防止模型无限循环调用工具。
        max_steps = self.max_steps
        # 用于向客户端累积透传 token / compaction 等元信息。
        metadata: Dict[Any, Any] = {}
        # 累积整个调用过程的 token、cost、compaction 等用量统计。
        stats = RequestStats()
        # 恢复流式调用时，iteration_offset 代表之前已经走过多少轮 LLM 调用；
        # 这里做非负校验，防止上层传入非法偏移导致计数错乱。
        if iteration_offset < 0:
            raise ValueError("iteration_offset must be non-negative")
        # i 表示当前已经完成的 LLM 轮次计数，初始化为恢复偏移量。
        i = iteration_offset

        # 主循环：每一轮做一次“必要压缩 -> 调 LLM -> 执行工具或结束”的闭环。
        while i < max_steps:
            # 支持外部中断；一旦用户取消，立即抛出专用异常终止本轮。
            if cancel_event and cancel_event.is_set():
                raise LLMInterruptedError()

            # 进入新的一轮 LLM 调用。
            i += 1
            logging.debug(f"running iteration {i}")

            # 到最后一轮时强制关闭工具调用，只允许模型给出最终答案。
            # 这样做是为了在接近 max_steps 上限时尽量收敛，而不是继续要求新工具。
            tools = None if i == max_steps else tools
            # 只要还有工具可用，就让模型自动决定是否要调工具；否则不传 tool_choice。
            tool_choice = "auto" if tools else None

            # 先做一次“是否需要压缩上下文”的预检查，必要时给前端一个开始压缩的事件。
            compaction_start_event = check_compaction_needed(self.llm, messages, tools)
            if compaction_start_event:
                yield compaction_start_event

            try:
                # 真正执行上下文压缩，确保消息和工具描述能塞进当前模型上下文窗口。
                limit_result = compact_if_necessary(
                    llm=self.llm, messages=messages, tools=tools
                )
            except CompactionInsufficientError as e:
                # 如果压缩后仍然塞不下，就把压缩阶段已经产生的事件先发出去，
                # 并把压缩消耗记入 stats，然后把异常继续抛给上层处理。
                yield from e.events
                if e.compaction_usage and e.compaction_usage.total_tokens > 0:
                    stats += e.compaction_usage
                raise

            # 把压缩过程里产生的所有事件继续流式输出。
            yield from limit_result.events
            # 用压缩后的消息替换当前上下文，后续 LLM 调用都基于这份精简结果。
            messages = limit_result.messages
            # 合并压缩阶段产生的元信息，保留已有 metadata 并叠加新值。
            metadata = metadata | limit_result.metadata

            # After compaction, emit a fresh token count so clients can update
            if limit_result.conversation_history_compacted:
                # 历史被压缩后，原有 token 统计已经过期，需要立刻补发一次最新 token 数。
                yield build_stream_event_token_count(
                    metadata={
                        "tokens": limit_result.tokens.model_dump(),
                        "max_tokens": limit_result.max_context_size,
                        "max_output_tokens": limit_result.maximum_output_token,
                    }
                )

            # Accumulate compaction costs
            # 把压缩本身消耗的 token/cost 计入总账，避免最终统计漏掉压缩成本。
            compaction = limit_result.compaction_usage
            if compaction and compaction.total_tokens > 0:
                # 一次 compact_if_necessary 视为一次 compaction。
                compaction.num_compactions = 1
                stats += compaction
                cost_logger.debug(
                    f"Compaction cost (streaming): ${compaction.total_cost:.6f} | "
                    f"Tokens: {compaction.prompt_tokens} prompt + {compaction.completion_tokens} completion = {compaction.total_tokens} total"
                )

            if (
                limit_result.conversation_history_compacted
                and RESET_REPEATED_TOOL_CALL_CHECK_AFTER_COMPACTION
            ):
                # 历史压缩后，重复调用检测可能失去一部分上下文。
                # 在开启该开关时，直接重置 tool_calls，避免基于不完整历史误判“重复调用”。
                tool_calls = []

            logging.debug(f"sending messages={messages}\n\ntools={tools}")

            try:
                # 注意这里不是底层 token 级流式，而是“每轮一个完整响应”的迭代流。
                # Holmes 自己把多轮 LLM + tool use 组织成事件流，便于统一控制审批、前端暂停等状态。
                full_response = self.llm.completion(
                    messages=parse_messages_tags(messages),  # type: ignore
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    temperature=TEMPERATURE,
                    stream=False,
                    drop_params=True,
                )

                # Accumulate cost information for this iteration
                # 把这一轮 LLM completion 的 usage/cost 提取出来并累计。
                response_stats = RequestStats.from_response(full_response)
                if response_stats.total_tokens > 0:
                    cost_logger.debug(
                        f"LLM iteration cost: ${response_stats.total_cost:.6f} | "
                        f"Tokens: {response_stats.prompt_tokens} prompt + {response_stats.completion_tokens} completion = {response_stats.total_tokens} total"
                    )
                elif response_stats.total_cost > 0:
                    cost_logger.debug(f"LLM iteration cost: ${response_stats.total_cost:.6f} | Token usage not available")
                if LOG_LLM_USAGE_RESPONSE:
                    usage = getattr(full_response, "usage", None)
                    if usage:
                        logging.info(f"LLM usage response:\n{usage}\n")
                stats += response_stats

            # catch a known error that occurs with Azure and replace the error message with something more obvious to the user
            except BadRequestError as e:
                # 对 Azure 某些不支持 tools 的模型给出更直白的报错，方便用户定位模型版本问题。
                if "Unrecognized request arguments supplied: tool_choice, tools" in str(
                    e
                ):
                    raise Exception(
                        "The Azure model you chose is not supported. Model version 1106 and higher required."
                    ) from e
                else:
                    logging.error(
                        f"LLM BadRequestError on model={self.llm.model} (streaming iteration {i}): {e}",
                        exc_info=True,
                    )
                    raise
            except Exception as e:
                # 其余异常直接记录完整上下文后抛出，交给上层统一处理。
                logging.error(
                    f"LLM call failed on model={self.llm.model} (streaming iteration {i}): "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise

            # completion 返回后再检查一次取消状态，避免在慢请求结束后继续推进后续逻辑。
            if cancel_event and cancel_event.is_set():
                raise LLMInterruptedError()

            # 取出本轮 assistant message，它可能包含普通文本，也可能包含 tool_calls。
            response_message = full_response.choices[0].message  # type: ignore

            # 先把 assistant 消息写回对话历史。
            # 这样无论后面是直接结束还是继续调工具，messages 都保持完整链路。
            messages.append(
                response_message.model_dump(
                    exclude_defaults=True, exclude_unset=True, exclude_none=True
                )
            )

            # 在 assistant 消息入历史后，立刻发一次最新 token 统计。
            yield self._emit_token_count(messages, tools, full_response, limit_result, metadata, stats)

            # 看本轮模型是否请求了工具调用。
            tools_to_call = getattr(response_message, "tool_calls", None)
            if not tools_to_call:
                # 没有工具调用，说明模型给出的就是最终回答，直接结束本次流。
                yield StreamMessage(
                    event=StreamEvents.ANSWER_END,
                    data={
                        "content": response_message.content,
                        "messages": messages,
                        "metadata": metadata,
                        "tool_calls": all_tool_calls,
                        "num_llm_calls": i,
                        "prompt": json.dumps(messages, indent=2),
                        "costs": stats.model_dump(),
                    },
                )
                return

            # 如果模型在发起工具前先输出了 reasoning 或普通文本，也要把它们通过 AI_MESSAGE 发给前端。
            reasoning = getattr(response_message, "reasoning_content", None)
            message = response_message.content
            if reasoning or message:
                yield StreamMessage(
                    event=StreamEvents.AI_MESSAGE,
                    data={
                        "content": message,
                        "reasoning": reasoning,
                        "metadata": metadata,
                    },
                )

            # Check if any tools require approval or are frontend-defined
            # 收集这批工具里需要用户审批的调用。
            pending_approvals = []
            # 收集这批工具里需要前端本地执行的调用。
            pending_frontend_calls: list[PendingFrontendToolCall] = []

            # Extract session approved prefixes from conversation history
            # 从既有消息历史里恢复本会话已批准的 bash 前缀，
            # 这样后续同前缀命令可以在同一轮或后续轮次中自动放行。
            session_prefixes = extract_bash_session_prefixes(messages)

            # 并发执行本轮所有工具调用，减少整体等待时间。
            # max_workers 取 16，是一个相对激进但受控的并行上限。
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = []
                for tool_index, t in enumerate(tools_to_call, 1):  # type: ignore
                    # 为每个工具生成全局递增的展示编号，方便 UI 侧追踪。
                    tool_number = tool_number_offset + tool_index

                    # 把工具提交给线程池异步执行；实际执行逻辑里会处理重复检测、审批、spill 等细节。
                    future = executor.submit(
                        self._invoke_llm_tool_call,
                        tool_to_call=t,  # type: ignore
                        previous_tool_calls=tool_calls,
                        trace_span=trace_span,
                        tool_number=tool_number,
                        session_approved_prefixes=session_prefixes,
                        request_context=request_context,
                        enable_tool_approval=enable_tool_approval,
                    )
                    futures.append(future)
                    # 工具一提交就先发 START_TOOL，让前端立刻显示“开始执行”状态，而不是等结果回来。
                    yield StreamMessage(
                        event=StreamEvents.START_TOOL,
                        data={"tool_name": t.function.name, "id": t.id},
                    )

                # 按“谁先完成谁先返回”的顺序消费工具结果，提升感知速度。
                for future in concurrent.futures.as_completed(futures):
                    if cancel_event and cancel_event.is_set():
                        # 一旦取消，尽量把尚未开始的 future 取消掉，然后中断整个流程。
                        for f in futures:
                            f.cancel()
                        raise LLMInterruptedError()

                    # 取出单个工具执行结果对象。
                    tool_call_result: ToolCallResult = future.result()

                    # 转成客户端友好的字典形式，后面多个分支都会用到。
                    tool_result_dict = tool_call_result.to_client_dict()

                    if (
                        tool_call_result.result.status
                        == StructuredToolResultStatus.APPROVAL_REQUIRED
                    ):
                        if enable_tool_approval:
                            # 开启审批模式时，不立即把这个工具结果喂回 LLM，
                            # 而是先登记为 pending approval，等待用户确认后再恢复执行。
                            pending_approvals.append(
                                PendingToolApproval(
                                    tool_call_id=tool_call_result.tool_call_id,
                                    tool_name=tool_call_result.tool_name,
                                    description=tool_call_result.description,
                                    params=tool_call_result.result.params or {},
                                )
                            )

                            # 虽然工具还没真正执行成功，但仍把“需要审批”的结果发给客户端，
                            # 这样 UI 能展示是哪条命令卡在审批上。
                            all_tool_calls.append(tool_result_dict)
                            yield StreamMessage(
                                event=StreamEvents.TOOL_RESULT,
                                data=tool_result_dict,
                            )
                        else:
                            # 未开启审批模式时，把“需要审批”直接降级成 ERROR，
                            # 避免模型误以为该工具已经执行成功。
                            tool_call_result.result.status = (
                                StructuredToolResultStatus.ERROR
                            )
                            tool_call_result.result.error = f"Tool call rejected for security reasons: {tool_call_result.result.error}"
                            tool_result_dict = tool_call_result.to_client_dict()

                            # 这种错误结果要写回本轮工具历史和消息历史，
                            # 让模型下一轮能看到“该工具因安全原因被拒绝”这个事实。
                            tool_calls.append(tool_result_dict)
                            all_tool_calls.append(tool_result_dict)
                            messages.append(tool_call_result.to_llm_message(supports_vision=self._supports_vision()))

                            yield StreamMessage(
                                event=StreamEvents.TOOL_RESULT,
                                data=tool_result_dict,
                            )

                    elif (
                        tool_call_result.result.status
                        == StructuredToolResultStatus.FRONTEND_PAUSE
                    ):
                        # Frontend tool — collect for pause, don't feed result to LLM
                        # 前端工具不能在后端直接执行，所以这里只记录待前端处理的信息，
                        # 暂时不把任何结果喂给 LLM。
                        pending_frontend_calls.append(
                            PendingFrontendToolCall(
                                tool_call_id=tool_call_result.tool_call_id,
                                tool_name=tool_call_result.tool_name,
                                arguments=tool_call_result.result.params or {},
                            )
                        )
                        # 给本轮工具历史保存一个简化对象，便于最终汇总和客户端展示。
                        frontend_call_dict = {
                            "tool_call_id": tool_call_result.tool_call_id,
                            "tool_name": tool_call_result.tool_name,
                            "name": tool_call_result.tool_name,
                        }
                        tool_calls.append(frontend_call_dict)
                        all_tool_calls.append(frontend_call_dict)

                    else:
                        # 正常工具结果：加入本轮历史、加入总历史，并写回 messages 供下一轮 LLM 使用。
                        tool_calls.append(tool_result_dict)
                        all_tool_calls.append(tool_result_dict)
                        messages.append(tool_call_result.to_llm_message())

                        # 同步把工具结果事件流给客户端。
                        yield StreamMessage(
                            event=StreamEvents.TOOL_RESULT,
                            data=tool_result_dict,
                        )

                # Emit updated token counts after tool results
                # 工具结果写回 messages 后，上下文体积发生变化，因此再补发一次 token 统计。
                yield self._emit_token_count(messages, tools, full_response, limit_result, metadata, stats)

                # Mark any pending frontend tool calls in assistant messages
                if pending_frontend_calls:
                    for fc in pending_frontend_calls:
                        # 在 assistant 原始 tool_call 请求上打标记，方便恢复时精确定位。
                        tool_call = self.find_assistant_tool_call_request(
                            tool_call_id=fc.tool_call_id, messages=messages
                        )
                        tool_call["pending_frontend"] = True

                # Mark any pending approval tool calls in assistant messages
                if pending_approvals:
                    for approval in pending_approvals:
                        # 审批暂停同样通过在原始 assistant tool_call 上打 pending_approval 标记来恢复。
                        tool_call = self.find_assistant_tool_call_request(
                            tool_call_id=approval.tool_call_id, messages=messages
                        )
                        tool_call["pending_approval"] = True

                # If either type of pause is needed, emit a single APPROVAL_REQUIRED
                # event that carries both pending_approvals and pending_frontend_tool_calls.
                # The client checks which lists are populated and handles accordingly.
                if pending_approvals or pending_frontend_calls:
                    # 不论是用户审批还是前端本地执行，本质都是“当前轮次必须暂停并等待外部动作”，
                    # 因此统一用 APPROVAL_REQUIRED 这个事件承载两类 pending 信息。
                    yield StreamMessage(
                        event=StreamEvents.APPROVAL_REQUIRED,
                        data={
                            "content": None,
                            "messages": messages,
                            "pending_approvals": [
                                approval.model_dump() for approval in pending_approvals
                            ],
                            "pending_frontend_tool_calls": [
                                fc.model_dump() for fc in pending_frontend_calls
                            ],
                            "num_llm_calls": i,
                            "costs": stats.model_dump(),
                        },
                    )
                    # 一旦进入暂停态，本次 call_stream 就先返回；待外部准备好结果后再恢复。
                    return

                # Update the tool number offset for the next iteration
                # 本轮工具都处理完后，推进全局工具编号偏移量，保证下一轮编号连续。
                tool_number_offset += len(tools_to_call)

                # Re-fetch tools if runbook was just activated (enables restricted tools)
                if self._runbook_in_use and tools is not None:
                    # 某些受限工具在 fetch_runbook 成功后才解锁；
                    # 因此这里重新取一次工具列表，让后续轮次能看到新开放的工具。
                    new_tools = self._get_tools()
                    if len(new_tools) != len(tools):
                        logging.info(
                            f"Runbook activated - refreshing tools list ({len(tools)} -> {len(new_tools)} tools)"
                        )
                        tools = new_tools

        # 走到这里说明模型连续多轮都没有收敛，超过了允许的最大步数。
        raise Exception(
            f"Too many LLM calls - exceeded max_steps: {i}/{self.max_steps}"
        )

    def find_assistant_tool_call_request(
        self, tool_call_id: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        for message in messages:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls", []):
                    if tool_call.get("id") == tool_call_id:
                        return tool_call

        # Should not happen unless there is a bug.
        # If we are here
        raise Exception(
            f"Failed to find assistant request for a tool_call in conversation history. tool_call_id={tool_call_id}"
        )
