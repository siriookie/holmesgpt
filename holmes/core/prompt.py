import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from holmes.plugins.prompts import load_and_render_prompt
from holmes.plugins.runbooks import RunbookCatalog
from holmes.utils.global_instructions import Instructions, generate_runbooks_args
from holmes.version import get_version


class PromptComponent(str, Enum):
    # User prompt components
    FILES = "files"
    TODOWRITE_REMINDER = "todowrite_reminder"
    TIME_RUNBOOKS = "time_runbooks"
    # System prompt components
    INTRO = "intro"
    ASK_USER = "ask_user"
    TODOWRITE_INSTRUCTIONS = "todowrite_instructions"
    AI_SAFETY = "ai_safety"
    TOOLSET_INSTRUCTIONS = "toolset_instructions"
    PERMISSION_ERRORS = "permission_errors"
    GENERAL_INSTRUCTIONS = "general_instructions"
    STYLE_GUIDE = "style_guide"
    CLUSTER_NAME = "cluster_name"
    SYSTEM_PROMPT_ADDITIONS = "system_prompt_additions"


# Components that are disabled by default (can be explicitly enabled via overrides or env var)
DISABLED_BY_DEFAULT = {PromptComponent.AI_SAFETY}


class InvalidImageDictError(ValueError):
    """Raised when an image dict is missing required keys or is malformed."""

    def __init__(self, provided_keys: List[str]):
        self.provided_keys = provided_keys
        super().__init__(
            f"Image dict must contain a 'url' key. Got keys: {provided_keys}"
        )


def build_vision_content(
    text: str, images: List[Union[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Build content array for vision models with text and images.

    Args:
        text: The text content
        images: List of images, each can be:
            - str: URL or base64 data URI
            - dict: Object with 'url' (required), 'detail', and 'format' fields

    Returns:
        List of content items in OpenAI vision format

    Raises:
        InvalidImageDictError: If an image dict is missing the 'url' key
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for image_item in images:
        if isinstance(image_item, str):
            content.append({"type": "image_url", "image_url": {"url": image_item}})
        else:
            if "url" not in image_item:
                raise InvalidImageDictError(list(image_item.keys()))
            image_url_obj: Dict[str, Any] = {"url": image_item["url"]}
            if "detail" in image_item:
                image_url_obj["detail"] = image_item["detail"]
            if "format" in image_item:
                image_url_obj["format"] = image_item["format"]
            content.append({"type": "image_url", "image_url": image_url_obj})
    return content


def is_prompt_allowed_by_env(component: PromptComponent) -> bool:
    """
    Check if a prompt component is allowed by the ENABLED_PROMPTS environment variable.

    Environment variable: ENABLED_PROMPTS
    - If not set: all prompts are ENABLED (production default)
    - If set to "none": all prompts are disabled
    - Comma-separated names (e.g., "files,ai_safety,time_runbooks")
    """
    enabled_prompts = os.environ.get("ENABLED_PROMPTS", "")

    if not enabled_prompts:
        return True  # Default: all enabled
    if enabled_prompts.lower() == "none":
        return False

    enabled_names = [x.strip().lower() for x in enabled_prompts.split(",")]
    return component.value in enabled_names


def is_component_enabled(
    component: PromptComponent,
    overrides: Optional[Dict[PromptComponent, bool]] = None,
) -> bool:
    """
    Check if a prompt component is enabled, considering both env var and API overrides.

    Precedence: env var > API override > default
    - If env var disables component: always disabled (API can't override)
    - If env var allows component: API override decides, or use default
    - Default is enabled for most components, except those in DISABLED_BY_DEFAULT
    """
    # 先检查环境变量层面的总开关。
    # 这是最高优先级，因为环境变量通常代表部署/运行时的强约束，
    # 需要能在不改代码的情况下统一禁止某些 prompt 组件。
    env_allowed = is_prompt_allowed_by_env(component)
    if not env_allowed:
        # 只要环境变量不允许，这个组件就必须关闭；
        # 即使调用方在 overrides 里显式打开，也不能越过运行时策略。
        return False  # env var wins, can't override to enabled
    if overrides and component in overrides:
        # 环境变量允许后，再看调用方有没有传入更细粒度的 API 覆盖。
        # 这样做的好处是：同一套默认行为下，不同调用场景仍可按需裁剪组件。
        return overrides[component]  # env allows, API decides
    # 当前两层都没有明确指定时，回落到默认策略：
    # 大多数组件默认开启，只有 DISABLED_BY_DEFAULT 里的组件默认关闭。
    # 这种设计能让常用能力默认可用，同时把更敏感或更昂贵的内容改成显式开启。
    return component not in DISABLED_BY_DEFAULT  # env allows, no override, use default


def append_file_to_user_prompt(user_prompt: str, file_path: Path) -> str:
    with file_path.open("r") as f:
        user_prompt += f"\n\n<attached-file path='{file_path.absolute()}'>\n{f.read()}\n</attached-file>"

    return user_prompt


def append_all_files_to_user_prompt(
    user_prompt: str,
    file_paths: Optional[List[Path]],
) -> str:
    if not file_paths:
        return user_prompt

    for file_path in file_paths:
        user_prompt = append_file_to_user_prompt(user_prompt, file_path)

    return user_prompt


def get_tasks_management_system_reminder() -> str:
    return (
        "\n\n<system-reminder>\nIMPORTANT: You have access to the TodoWrite tool. It creates a TodoList, in order to track progress. It's very important. You MUST use it:\n1. FIRST: Ask your self which sub problems you need to solve in order to answer the question."
        "Do this, BEFORE any other tools\n2. "
        "AFTER EVERY TOOL CALL: If required, update the TodoList\n3. "
        "\n\nFAILURE TO UPDATE TodoList = INCOMPLETE INVESTIGATION\n\n"
        "Example flow:\n- Think and divide to sub problems → create TodoList → Perform each task on the list → Update list → Verify your solution\n</system-reminder>"
    )


def _has_content(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip())


def _should_enable_runbooks(context: Dict[str, str]) -> bool:
    return any(
        (
            _has_content(context.get("runbook_catalog")),
            _has_content(context.get("custom_instructions")),
            _has_content(context.get("global_instructions")),
        )
    )


def generate_user_prompt(
    user_prompt: str,
    context: Dict[str, str],
) -> str:
    runbooks_enabled = _should_enable_runbooks(context)

    return load_and_render_prompt(
        "builtin://base_user_prompt.jinja2",
        context={
            "user_prompt": user_prompt,
            "runbooks_enabled": runbooks_enabled,
            **context,
        },
    )


def build_system_prompt(
    toolsets: List[Any],
    runbooks: Optional[RunbookCatalog],
    system_prompt_additions: Optional[str],
    cluster_name: Optional[str],
    ask_user_enabled: bool,
    prompt_component_overrides: Dict[PromptComponent, bool],
) -> Optional[str]:
    """
    Build the system prompt for both CLI and server modes.
    Returns None if the rendered prompt is empty.
    """

    def is_enabled(component: PromptComponent) -> bool:
        # 统一走同一个开关判断入口，避免下面重复传 overrides，
        # 也让系统提示词各组件的启停规则保持一致。
        return is_component_enabled(component, prompt_component_overrides)

    # 先单独算出 toolset instructions 是否启用，后面会同时影响：
    # 1. 模板里的布尔开关
    # 2. 是否真的把 toolsets 明细传给模板
    # 这样做是为了避免“组件关闭了，但数据还传进模板”带来的歧义和无效上下文。
    toolset_instructions_enabled = is_enabled(PromptComponent.TOOLSET_INSTRUCTIONS)

    # 构造模板上下文，交给 generic_ask.jinja2 统一渲染系统提示词。
    # 这里大部分字段都拆成 “是否启用” + “真实内容” 两层，
    # 原因是模板更适合根据布尔开关决定是否渲染某个 section，
    # Python 侧则负责准备好干净、裁剪后的输入数据。
    template_context = {
        # 注入当前 Holmes 版本，方便模板在需要时输出版本相关说明。
        "holmes_version": get_version(),
        # 控制系统提示词中的基础介绍部分是否出现。
        "intro_enabled": is_enabled(PromptComponent.INTRO),
        # ask_user 既受调用方能力控制，也受 prompt 组件开关控制；
        # 两者都满足时才启用，避免 server/CLI 能力与模板文案不一致。
        "ask_user_enabled": ask_user_enabled and is_enabled(PromptComponent.ASK_USER),
        # 控制是否向模型注入 TodoWrite 的使用规范。
        "todowrite_enabled": is_enabled(PromptComponent.TODOWRITE_INSTRUCTIONS),
        # AI safety 默认可能是关闭的，所以这里显式带上开关结果。
        "ai_safety_enabled": is_enabled(PromptComponent.AI_SAFETY),
        # 告诉模板是否要渲染 toolset 说明区块。
        "toolset_instructions_enabled": toolset_instructions_enabled,
        # 控制权限错误相关的额外指导是否出现。
        "permission_errors_enabled": is_enabled(PromptComponent.PERMISSION_ERRORS),
        # 控制通用行为规范说明是否出现。
        "general_instructions_enabled": is_enabled(
            PromptComponent.GENERAL_INSTRUCTIONS
        ),
        # 控制输出风格指南是否注入到系统提示词中。
        "style_guide_enabled": is_enabled(PromptComponent.STYLE_GUIDE),
        # runbooks 既要“对象存在且有内容”，又要组件开关允许，才值得在模板中启用。
        # 这样做是为了避免模板渲染出空的 runbook 区块，徒增上下文噪音。
        "runbooks_enabled": bool(runbooks and getattr(runbooks, "catalog", True))
        and is_enabled(PromptComponent.TIME_RUNBOOKS),
        # cluster_name 关闭时直接传 None，而不是传原值，
        # 这样模板可以更简单地用“是否为空”判断是否渲染该信息。
        "cluster_name": cluster_name
        if is_enabled(PromptComponent.CLUSTER_NAME)
        else None,
        # 只有启用了 toolset instructions 才把 toolsets 列表真正传入模板，
        # 原因是这些内容通常较长，关闭时应当一并裁掉，减少无效 token 消耗。
        "toolsets": toolsets if toolset_instructions_enabled else [],
        # 额外 system prompt 补充说明同样受组件开关控制；
        # 关闭时传空字符串，避免旧调用方传值后仍被意外拼进最终提示词。
        "system_prompt_additions": system_prompt_additions
        if is_enabled(PromptComponent.SYSTEM_PROMPT_ADDITIONS)
        else "",
    }

    # 使用统一模板渲染最终系统提示词。
    result = load_and_render_prompt("builtin://generic_ask.jinja2", template_context)
    # 如果模板最终产物为空白，就返回 None 而不是空字符串。
    # 这样上层在组装 messages 时可以直接跳过 system role，逻辑更干净。
    return result if result and result.strip() else None


UserPromptContent = Union[str, List[Dict[str, Any]]]


def build_user_prompt(
    user_prompt: str,
    runbooks: Optional[RunbookCatalog],
    global_instructions: Optional[Instructions],
    file_paths: Optional[List[Path]],
    include_todowrite_reminder: bool,
    images: Optional[List[Union[str, Dict[str, Any]]]],
    prompt_component_overrides: Dict[PromptComponent, bool],
) -> UserPromptContent:
    """Build the user prompt with all enrichments.

    Returns:
        Either a string or a list of content dicts (for vision models with images).
    """

    def is_enabled(component: PromptComponent) -> bool:
        return is_component_enabled(component, prompt_component_overrides)

    if file_paths and is_enabled(PromptComponent.FILES):
        user_prompt = append_all_files_to_user_prompt(user_prompt, file_paths)

    if include_todowrite_reminder and is_enabled(PromptComponent.TODOWRITE_REMINDER):
        user_prompt += get_tasks_management_system_reminder()

    if is_enabled(PromptComponent.TIME_RUNBOOKS):
        runbooks_ctx = generate_runbooks_args(
            runbook_catalog=runbooks,
            global_instructions=global_instructions,
        )
        user_prompt = generate_user_prompt(user_prompt, runbooks_ctx)

    if images:
        return build_vision_content(user_prompt, images)
    return user_prompt


def build_prompts(
    toolsets: List[Any],
    user_prompt: str,
    runbooks: Optional[RunbookCatalog],
    global_instructions: Optional[Instructions],
    system_prompt_additions: Optional[str],
    cluster_name: Optional[str],
    ask_user_enabled: bool,
    file_paths: Optional[List[Path]],
    include_todowrite_reminder: bool,
    images: Optional[List[Union[str, Dict[str, Any]]]],
    prompt_component_overrides: Optional[Dict[PromptComponent, bool]] = None,
) -> Tuple[Optional[str], UserPromptContent]:
    """Build both system and user prompts."""
    if prompt_component_overrides is None:
        prompt_component_overrides = {}

    system_prompt = build_system_prompt(
        toolsets=toolsets,
        runbooks=runbooks,
        system_prompt_additions=system_prompt_additions,
        cluster_name=cluster_name,
        ask_user_enabled=ask_user_enabled,
        prompt_component_overrides=prompt_component_overrides,
    )
    user_content = build_user_prompt(
        user_prompt=user_prompt,
        runbooks=runbooks,
        global_instructions=global_instructions,
        file_paths=file_paths,
        include_todowrite_reminder=include_todowrite_reminder,
        images=images,
        prompt_component_overrides=prompt_component_overrides,
    )
    return system_prompt, user_content


def build_initial_ask_messages(
    initial_user_prompt: str,
    file_paths: Optional[List[Path]],
    tool_executor: Any,  # ToolExecutor type
    runbooks: Optional[RunbookCatalog] = None,
    system_prompt_additions: Optional[str] = None,
    global_instructions: Optional[Instructions] = None,
    cluster_name: Optional[str] = None,
    prompt_component_overrides: Optional[Dict[PromptComponent, bool]] = None,
) -> List[Dict]:
    """Build the initial messages for the CLI ask command."""
    system_prompt, user_prompt = build_prompts(
        toolsets=tool_executor.toolsets,
        user_prompt=initial_user_prompt,
        runbooks=runbooks,
        global_instructions=global_instructions,
        system_prompt_additions=system_prompt_additions,
        cluster_name=cluster_name,
        ask_user_enabled=True,
        file_paths=file_paths,
        include_todowrite_reminder=True,
        images=None,
        prompt_component_overrides=prompt_component_overrides,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages
