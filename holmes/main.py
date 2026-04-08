# ruff: noqa: E402
import os

from holmes.utils.cert_utils import add_custom_certificate

ADDITIONAL_CERTIFICATE: str = os.environ.get("CERTIFICATE", "")
if add_custom_certificate(ADDITIONAL_CERTIFICATE):
    print("added custom certificate")

# DO NOT ADD ANY IMPORTS OR CODE ABOVE THIS LINE
# IMPORTING ABOVE MIGHT INITIALIZE AN HTTPS CLIENT THAT DOESN'T TRUST THE CUSTOM CERTIFICATE
import sys
from holmes.utils.colors import USER_COLOR
import json
import logging
import socket
import uuid
from pathlib import Path
from typing import List, Optional

import typer
from rich.markdown import Markdown
from rich.rule import Rule

from holmes import get_version  # type: ignore
from holmes.config import (
    Config,
    SourceFactory,
    SupportedTicketSources,
)
from holmes.core.prompt import (
    PromptComponent,
    build_initial_ask_messages,
    build_system_prompt,
    generate_user_prompt,
)
from holmes.core.resource_instruction import ResourceInstructionDocument
from holmes.core.tool_calling_llm import LLMResult, ToolCallingLLM
from holmes.core.tools import pretty_print_toolset_status
from holmes.core.tools_utils.filesystem_result_storage import tool_result_storage
from holmes.core.tracing import SpanType, TracingFactory
from holmes.interactive import InitProgressRenderer, run_interactive_loop, silence_display_loggers
from holmes.plugins.destinations import DestinationType
from holmes.plugins.interfaces import Issue
from holmes.plugins.prompts import load_and_render_prompt
from holmes.plugins.sources.opsgenie import OPSGENIE_TEAM_INTEGRATION_KEY_HELP
from holmes.utils.console.logging import init_logging
from holmes.utils.console.result import handle_result
from holmes.utils.file_utils import write_json_file
from holmes.checks.checks_cli import checks_app
from holmes.common.cli_commons import (
    opt_api_key,
    opt_config_file,
    opt_model,
    opt_verbose,
)
from holmes.toolset_config_tui import run_toolset_config_tui

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)



investigate_app = typer.Typer(
    add_completion=False,
    name="investigate",
    no_args_is_help=True,
    help="Investigate firing alerts or tickets",
)
app.add_typer(investigate_app, name="investigate")
generate_app = typer.Typer(
    add_completion=False,
    name="generate",
    no_args_is_help=True,
    help="Generate new integrations or test data",
)
app.add_typer(generate_app, name="generate")
toolset_app = typer.Typer(
    add_completion=False,
    name="toolset",
    no_args_is_help=True,
    help="Toolset management commands",
)
app.add_typer(toolset_app, name="toolset")

app.add_typer(checks_app, name="checks")


# Common cli options defined in holmes.common.cli_commons:
# opt_api_key, opt_model, opt_config_file, opt_verbose
# The defaults for options that are also in the config file MUST be None or else the cli defaults will override settings in the config file
opt_fast_model: Optional[str] = typer.Option(
    None, help="Optional fast model for summarization tasks"
)
opt_custom_toolsets: Optional[List[Path]] = typer.Option(
    [],
    "--custom-toolsets",
    "-t",
    help="Path to a custom toolsets. The status of the custom toolsets specified here won't be cached (can specify -t multiple times to add multiple toolsets)",
)
opt_max_steps: Optional[int] = typer.Option(
    100,
    "--max-steps",
    help="Advanced. Maximum number of steps the LLM can take to investigate the issue",
)
opt_log_costs: bool = typer.Option(
    False,
    "--log-costs",
    help="Show LLM cost information in the output",
)
opt_echo_request: bool = typer.Option(
    True,
    "--echo/--no-echo",
    help="Echo back the question provided to HolmesGPT in the output",
)
opt_destination: Optional[DestinationType] = typer.Option(
    DestinationType.CLI,
    "--destination",
    help="Destination for the results of the investigation (defaults to STDOUT)",
)
opt_slack_token: Optional[str] = typer.Option(
    None,
    "--slack-token",
    help="Slack API key if --destination=slack (experimental). Can generate with `pip install robusta-cli && robusta integrations slack`",
)
opt_slack_channel: Optional[str] = typer.Option(
    None,
    "--slack-channel",
    help="Slack channel if --destination=slack (experimental). E.g. #devops",
)
opt_json_output_file: Optional[str] = typer.Option(
    None,
    "--json-output-file",
    help="Save the complete output in json format in to a file",
    envvar="HOLMES_JSON_OUTPUT_FILE",
)

opt_documents: Optional[str] = typer.Option(
    None,
    "--documents",
    help="Additional documents to provide the LLM (typically URLs to runbooks)",
)


def parse_documents(documents: Optional[str]) -> List[ResourceInstructionDocument]:
    resource_documents = []

    if documents is not None:
        data = json.loads(documents)
        for item in data:
            resource_document = ResourceInstructionDocument(**item)
            resource_documents.append(resource_document)

    return resource_documents


def _investigate_issue(
    ai: ToolCallingLLM,
    issue: Issue,
    config: Config,
) -> LLMResult:
    """Investigate an issue using the standard ask system prompt with investigation additions."""
    investigation_additions = f"Provide a terse analysis of the following {issue.source_type} alert/issue and why it is firing."
    system_prompt = build_system_prompt(
        toolsets=ai.tool_executor.toolsets,
        runbooks=None,
        system_prompt_additions=investigation_additions,
        cluster_name=config.cluster_name,
        ask_user_enabled=False,
        prompt_component_overrides={},
    )
    user_prompt = generate_user_prompt(
        f"\n #This is context from the issue:\n{issue.raw}",
        context={},
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return ai.call(messages)


# TODO: add streaming output
@app.command()
def ask(
    # 用户直接在命令行里输入的问题正文。
    prompt: Optional[str] = typer.Argument(
        None, help="What to ask the LLM (user prompt)"
    ),
    # 允许从文件读取 prompt，适合长文本输入。
    prompt_file: Optional[Path] = typer.Option(
        None,
        "--prompt-file",
        "-pf",
        help="File containing the prompt to ask the LLM (overrides the prompt argument if provided)",
    ),
    # ask 命令复用的通用配置项。
    api_key: Optional[str] = opt_api_key,
    model: Optional[str] = opt_model,
    fast_model: Optional[str] = opt_fast_model,
    config_file: Optional[Path] = opt_config_file,
    custom_toolsets: Optional[List[Path]] = opt_custom_toolsets,
    max_steps: Optional[int] = opt_max_steps,
    verbose: Optional[List[bool]] = opt_verbose,
    log_costs: bool = opt_log_costs,
    # ask 命令常用但并非所有命令共享的参数。
    destination: Optional[DestinationType] = opt_destination,
    slack_token: Optional[str] = opt_slack_token,
    slack_channel: Optional[str] = opt_slack_channel,
    # 调试时可展示每个工具调用的原始输出。
    show_tool_output: bool = typer.Option(
        False,
        "--show-tool-output",
        help="Advanced. Show the output of each tool that was called",
    ),
    # 把本地文件内容附加到模型上下文中。
    include_file: Optional[List[Path]] = typer.Option(
        [],
        "--file",
        "-f",
        help="File to append to prompt (can specify -f multiple times to add multiple files)",
    ),
    json_output_file: Optional[str] = opt_json_output_file,
    echo_request: bool = opt_echo_request,
    # 是否进入交互模式；脚本执行时通常会关闭。
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        "-i/-n",
        help="Enter interactive mode after the initial question? For scripting, disable this with --no-interactive",
    ),
    # 是否强制刷新 toolset 状态缓存。
    refresh_toolsets: bool = typer.Option(
        False,
        "--refresh-toolsets",
        help="Refresh the toolsets status",
    ),
    # 把本次执行接入 tracing 后端，便于排查链路。
    trace: Optional[str] = typer.Option(
        None,
        "--trace",
        help="Enable tracing to the specified provider (e.g., 'braintrust')",
    ),
    # 额外附加到 system prompt 的说明。
    system_prompt_additions: Optional[str] = typer.Option(
        None,
        "--system-prompt-additions",
        help="Additional content to append to the system prompt",
    ),
    # 自动拒绝所有未允许的 bash 调用。
    bash_always_deny: bool = typer.Option(
        False,
        "--bash-always-deny",
        help="Auto-deny all bash commands not in allow list without prompting",
    ),
    # 自动允许 bash 调用，适合已知安全的环境。
    bash_always_allow: bool = typer.Option(
        False,
        "--bash-always-allow",
        help="Bypass bash command approval checks. Recommended only for sandboxed environments",
    ),
    # 快速模式下跳过 TodoWrite 相关提示。
    fast_mode: bool = typer.Option(
        False,
        "--fast-mode",
        help="Skip TodoWrite planning phase for faster responses",
    ),
):
    """
    Ask any question and answer using available tools
    """
    # 这两个开关语义相反，所以必须在入口处做互斥校验。
    # 原因是如果两者同时为真，后续执行层无法判断 bash 到底该统一放行还是统一拒绝。
    if bash_always_deny and bash_always_allow:
        raise typer.BadParameter(
            "--bash-always-deny and --bash-always-allow are mutually exclusive. Choose one."
        )

    # 初始化控制台日志输出；后续所有 CLI 展示都依赖这个 console。
    console = init_logging(verbose, log_costs)  # type: ignore
    # 用于保存通过 stdin 管道传入的文本内容。
    piped_data = None

    # PyCharm 调试时 `isatty()` 可能返回 False，但此时并不一定真的有管道输入。
    # 如果直接 `read()`，调试过程可能卡住，所以要把这个场景排除掉。
    running_from_pycharm = os.environ.get("PYCHARM_HOSTED", False)

    # 非 TTY 且不是 PyCharm 调试时，通常意味着输入来自管道。
    if not sys.stdin.isatty() and not running_from_pycharm:
        piped_data = sys.stdin.read().strip()
        # 管道输入会先消费 stdin，因此不再适合进入交互模式。
        # 原因是交互模式后续还要继续读 stdin，容易出现无输入可读或行为异常。
        if interactive:
            console.print(
                "[bold yellow]Interactive mode disabled when reading piped input[/bold yellow]"
            )
            interactive = False

    # Silence display loggers early for interactive mode so that
    # init messages are rendered via the InitProgressRenderer instead.
    if interactive:
        # 提前静默这些 logger，避免它们和 Rich 的实时渲染争抢终端输出。
        # 这么做能减少交互模式下的闪烁、错位和重复输出。
        silence_display_loggers()

    # 把配置文件、环境变量和 CLI 参数合并成最终运行配置。
    config = Config.load_from_file(
        config_file,
        api_key=api_key,
        model=model,
        fast_model=fast_model,
        max_steps=max_steps,
        custom_toolsets_from_cli=custom_toolsets,
        slack_token=slack_token,
        slack_channel=slack_channel,
    )

    # 创建 tracer；即使没有配置 provider，也让后续逻辑始终走统一接口。
    tracer = TracingFactory.create_tracer(trace, project="HolmesGPT-CLI")
    # 开始一次新的实验/会话，用来聚合这次 ask 的完整链路。
    tracer.start_experiment()

    # prompt 和 prompt_file 代表同一份输入来源，只允许二选一。
    if prompt_file and prompt:
        raise typer.BadParameter(
            "You cannot provide both a prompt argument and a prompt file. Please use one or the other."
        )
    elif prompt_file:
        # 先验证文件存在，尽早给出明确错误。
        if not prompt_file.is_file():
            raise typer.BadParameter(f"Prompt file not found: {prompt_file}")
        # 读取文件内容作为最终 prompt。
        with prompt_file.open("r") as f:
            prompt = f.read()
        # 显示提示，帮助用户确认当前实际使用的是文件里的内容。
        console.print(
            f"[bold yellow]Loaded prompt from file {prompt_file}[/bold yellow]"
        )
    # 非交互模式下如果没有 prompt，也没有管道输入，就没有可执行内容。
    elif not prompt and not interactive and not piped_data:
        raise typer.BadParameter(
            "Either the 'prompt' argument or the --prompt-file option must be provided (unless using --interactive mode)."
        )

    # 把 stdin 输入包装进 prompt，让模型知道这是待分析的上下文文本。
    if piped_data:
        if prompt:
            # 如果用户同时给了问题和管道输入，就把两者拼接起来。
            prompt = f"Here's some piped output:\n\n{piped_data}\n\n{prompt}"
        else:
            # 只有管道输入时补一个默认问题，避免模型缺少明确任务指令。
            prompt = f"Here's some piped output:\n\n{piped_data}\n\nWhat can you tell me about this output?"

    # 非交互模式下可以把最终请求回显出来，方便脚本日志回放。
    if echo_request and not interactive and prompt:
        console.print(f"[bold {USER_COLOR}]User:[/bold {USER_COLOR}] {prompt}")

    # 默认不覆盖 prompt 组件。
    prompt_component_overrides = None
    if fast_mode:
        # 快速模式关闭 TodoWrite 相关提示，减少规划步骤带来的额外 token 和等待时间。
        prompt_component_overrides = {
            PromptComponent.TODOWRITE_INSTRUCTIONS: False,
            PromptComponent.TODOWRITE_REMINDER: False,
        }

    # 为本轮 ask 创建临时工具结果目录，退出 `with` 时会自动清理。
    with tool_result_storage() as tool_results_dir:
        # 只有交互模式会用到初始化阶段的进度渲染器。
        init_renderer = None
        on_event = None
        if interactive:
            # 在创建 LLM 前先显示初始化进度，让用户知道程序正在准备工具和模型。
            init_renderer = InitProgressRenderer(
                console, model_name=model or config.model or ""
            )
            # 把初始化事件回调交给后续组件，用于更新进度显示。
            on_event = init_renderer.on_event
            init_renderer.start()

        # 创建真正执行 ask 的 ToolCallingLLM 实例。
        ai = config.create_console_toolcalling_llm(
            dal=None,  # type: ignore
            refresh_toolsets=refresh_toolsets,  # flag to refresh the toolset status
            tracer=tracer,
            model_name=model,
            tool_results_dir=tool_results_dir,
            on_event=on_event,
        )

        if init_renderer is not None:
            # 初始化完成后及时停止渲染器，避免继续占用终端渲染区域。
            init_renderer.stop()

        if interactive:
            # 交互模式下把控制权交给 REPL 循环，后面的非交互分支不再执行。
            run_interactive_loop(
                ai,
                console,
                prompt,
                include_file,
                show_tool_output,
                tracer,
                config.get_runbook_catalog(),
                system_prompt_additions,
                json_output_file=json_output_file,
                bash_always_deny=bash_always_deny,
                bash_always_allow=bash_always_allow,
                prompt_component_overrides=prompt_component_overrides,
                config=config,
                config_file_path=config_file,
            )
            return

        if include_file:
            for file_path in include_file:
                # 让用户看到哪些文件会进入上下文，方便确认输入范围是否符合预期。
                console.print(
                    f"[bold yellow]Adding file {file_path} to context[/bold yellow]"
                )

        # 构造首轮 ask 的消息列表，内部会合并 prompt、附加文件、tool 信息和 runbook。
        messages = build_initial_ask_messages(
            prompt,  # type: ignore
            include_file,
            ai.tool_executor,
            config.get_runbook_catalog(),
            system_prompt_additions,
            prompt_component_overrides=prompt_component_overrides,
        )

        # 用 trace span 包裹整个 ask 调用，记录输入、输出和链路元数据。
        with tracer.start_trace(
            f'holmes ask "{prompt}"', span_type=SpanType.TASK
        ) as trace_span:
            # 记录用户输入，便于事后复盘这次调用的上下文。
            trace_span.log(input=prompt, metadata={"type": "user_question"})
            # 调用模型执行完整的工具推理流程。
            response = ai.call(messages, trace_span=trace_span)
            # 把最终输出也记入 trace，形成完整闭环。
            trace_span.log(
                output=response.result,
            )
            # 如果 tracing 后端支持外链，这里取回可展示的地址。
            trace_url = tracer.get_trace_url()

        # 用响应中的完整消息历史覆盖本地变量，便于后续保存完整上下文。
        messages = response.messages  # type: ignore # Update messages with the full history

        if json_output_file:
            # 用户要求导出 JSON 时，写出完整结构化响应结果。
            write_json_file(json_output_file, response.model_dump())

        # 把 ask 结果包装成统一的 Issue 对象，复用已有的结果处理链路。
        # 这样 ask / investigate 虽然入口不同，但展示与下游投递逻辑可以保持一致。
        issue = Issue(
            id=str(uuid.uuid4()),
            name=prompt,  # type: ignore
            source_type="holmes-ask",
            raw={"prompt": prompt, "full_conversation": messages},
            source_instance_id=socket.gethostname(),
        )
        # 统一处理最终输出，包含终端展示、目标投递以及可选的工具输出信息。
        handle_result(
            response,
            console,
            destination,  # type: ignore
            config,
            issue,
            show_tool_output,
            False,  # type: ignore
            log_costs,
        )

        if trace_url:
            console.print(f"🔍 View trace: {trace_url}")


@investigate_app.command()
def alertmanager(
    alertmanager_url: Optional[str] = typer.Option(None, help="AlertManager url"),
    alertmanager_alertname: Optional[str] = typer.Option(
        None,
        help="Investigate all alerts with this name (can be regex that matches multiple alerts). If not given, defaults to all firing alerts",
    ),
    alertmanager_label: Optional[List[str]] = typer.Option(
        [],
        help="For filtering alerts with a specific label. Must be of format key=value. If --alertmanager-label is passed multiple times, alerts must match ALL labels",
    ),
    alertmanager_username: Optional[str] = typer.Option(
        None, help="Username to use for basic auth"
    ),
    alertmanager_password: Optional[str] = typer.Option(
        None, help="Password to use for basic auth"
    ),
    alertmanager_file: Optional[Path] = typer.Option(
        None, help="Load alertmanager alerts from a file (used by the test framework)"
    ),
    alertmanager_limit: Optional[int] = typer.Option(
        None, "-n", help="Limit the number of alerts to process"
    ),
    # common options
    api_key: Optional[str] = opt_api_key,
    model: Optional[str] = opt_model,
    config_file: Optional[Path] = opt_config_file,  # type: ignore
    custom_toolsets: Optional[List[Path]] = opt_custom_toolsets,

    max_steps: Optional[int] = opt_max_steps,
    verbose: Optional[List[bool]] = opt_verbose,
    # advanced options for this command
    destination: Optional[DestinationType] = opt_destination,
    slack_token: Optional[str] = opt_slack_token,
    slack_channel: Optional[str] = opt_slack_channel,
    json_output_file: Optional[str] = opt_json_output_file,
):
    """
    Investigate a Prometheus/Alertmanager alert
    """
    console = init_logging(verbose)

    config = Config.load_from_file(
        config_file,
        api_key=api_key,
        model=model,
        max_steps=max_steps,
        alertmanager_url=alertmanager_url,
        alertmanager_username=alertmanager_username,
        alertmanager_password=alertmanager_password,
        alertmanager_alertname=alertmanager_alertname,
        alertmanager_label=alertmanager_label,
        alertmanager_file=alertmanager_file,
        slack_token=slack_token,
        slack_channel=slack_channel,
        custom_toolsets_from_cli=custom_toolsets,
    )

    with tool_result_storage() as tool_results_dir:
        ai = config.create_console_toolcalling_llm(model_name=model, tool_results_dir=tool_results_dir)

        source = config.create_alertmanager_source()

        try:
            issues = source.fetch_issues()
        except Exception as e:
            logging.error("Failed to fetch issues from alertmanager", exc_info=e)
            return

        if alertmanager_limit is not None:
            console.print(
                f"[bold yellow]Limiting to {alertmanager_limit}/{len(issues)} issues.[/bold yellow]"
            )
            issues = issues[:alertmanager_limit]

        if alertmanager_alertname is not None:
            console.print(
                f"[bold yellow]Analyzing {len(issues)} issues matching filter.[/bold yellow] [red]Press Ctrl+C to stop.[/red]"
            )
        else:
            console.print(
                f"[bold yellow]Analyzing all {len(issues)} issues. (Use --alertmanager-alertname to filter.)[/bold yellow] [red]Press Ctrl+C to stop.[/red]"
            )
        results = []
        for i, issue in enumerate(issues):
            console.print(
                f"[bold yellow]Analyzing issue {i+1}/{len(issues)}: {issue.name}...[/bold yellow]"
            )
            result = _investigate_issue(ai, issue, config)
            results.append({"issue": issue.model_dump(), "result": result.model_dump()})
            handle_result(result, console, destination, config, issue, False, True)  # type: ignore

        if json_output_file:
            write_json_file(json_output_file, results)


@generate_app.command("alertmanager-tests")
def generate_alertmanager_tests(
    alertmanager_url: Optional[str] = typer.Option(None, help="AlertManager url"),
    alertmanager_username: Optional[str] = typer.Option(
        None, help="Username to use for basic auth"
    ),
    alertmanager_password: Optional[str] = typer.Option(
        None, help="Password to use for basic auth"
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Path to dump alertmanager alerts as json (if not given, output curl commands instead)",
    ),
    config_file: Optional[Path] = opt_config_file,  # type: ignore
    verbose: Optional[List[bool]] = opt_verbose,
):
    """
    Connect to alertmanager and dump all alerts as either a json file or curl commands to simulate the alert (depending on --output flag)
    """
    console = init_logging(verbose)  # type: ignore
    config = Config.load_from_file(
        config_file,
        alertmanager_url=alertmanager_url,
        alertmanager_username=alertmanager_username,
        alertmanager_password=alertmanager_password,
    )

    source = config.create_alertmanager_source()
    if output is None:
        source.output_curl_commands(console)
    else:
        source.dump_raw_alerts_to_file(output)


@investigate_app.command()
def jira(
    jira_url: Optional[str] = typer.Option(
        None,
        help="Jira url - e.g. https://your-company.atlassian.net",
        envvar="JIRA_URL",
    ),
    jira_username: Optional[str] = typer.Option(
        None,
        help="The email address with which you log into Jira",
        envvar="JIRA_USERNAME",
    ),
    jira_api_key: str = typer.Option(
        None,
        envvar="JIRA_API_KEY",
    ),
    jira_query: Optional[str] = typer.Option(
        None,
        help="Investigate tickets matching a JQL query (e.g. 'project=DEFAULT_PROJECT')",
    ),
    update: Optional[bool] = typer.Option(False, help="Update Jira with AI results"),
    # common options
    api_key: Optional[str] = opt_api_key,
    model: Optional[str] = opt_model,
    config_file: Optional[Path] = opt_config_file,  # type: ignore
    custom_toolsets: Optional[List[Path]] = opt_custom_toolsets,

    max_steps: Optional[int] = opt_max_steps,
    verbose: Optional[List[bool]] = opt_verbose,
    json_output_file: Optional[str] = opt_json_output_file,
):
    """
    Investigate a Jira ticket
    """
    console = init_logging(verbose)

    config = Config.load_from_file(
        config_file,
        api_key=api_key,
        model=model,
        max_steps=max_steps,
        jira_url=jira_url,
        jira_username=jira_username,
        jira_api_key=jira_api_key,
        jira_query=jira_query,
        custom_toolsets_from_cli=custom_toolsets,
    )
    source = config.create_jira_source()
    try:
        issues = source.fetch_issues()
    except Exception as e:
        logging.error("Failed to fetch issues from Jira", exc_info=e)
        return

    console.print(
        f"[bold yellow]Analyzing {len(issues)} Jira tickets.[/bold yellow] [red]Press Ctrl+C to stop.[/red]"
    )

    results = []
    with tool_result_storage() as tool_results_dir:
        ai = config.create_console_toolcalling_llm(model_name=model, tool_results_dir=tool_results_dir)
        for i, issue in enumerate(issues):
            console.print(
                f"[bold yellow]Analyzing Jira ticket {i+1}/{len(issues)}: {issue.name}...[/bold yellow]"
            )
            result = _investigate_issue(ai, issue, config)

            console.print(Rule())
            console.print(f"[bold green]AI analysis of {issue.url}[/bold green]")
            console.print(Markdown(result.result.replace("\n", "\n\n")), style="bold green")  # type: ignore
            console.print(Rule())
            if update:
                source.write_back_result(issue.id, result)
                console.print(f"[bold]Updated ticket {issue.url}.[/bold]")
            else:
                console.print(
                    f"[bold]Not updating ticket {issue.url}. Use the --update option to do so.[/bold]"
                )

            results.append({"issue": issue.model_dump(), "result": result.model_dump()})

    if json_output_file:
        write_json_file(json_output_file, results)


# Define supported sources


@investigate_app.command()
def ticket(
    prompt: str = typer.Argument(help="What to ask the LLM (user prompt)"),
    source: SupportedTicketSources = typer.Option(
        ...,
        help=f"Source system to investigate the ticket from. Supported sources: {', '.join(s.value for s in SupportedTicketSources)}",
    ),
    ticket_url: Optional[str] = typer.Option(
        None,
        help="URL - e.g. https://your-company.atlassian.net",
        envvar="TICKET_URL",
    ),
    ticket_username: Optional[str] = typer.Option(
        None,
        help="The email address with which you log into your Source",
        envvar="TICKET_USERNAME",
    ),
    ticket_api_key: Optional[str] = typer.Option(
        None,
        envvar="TICKET_API_KEY",
    ),
    ticket_id: Optional[str] = typer.Option(
        None,
        help="ticket ID to investigate (e.g., 'KAN-1')",
    ),
    update: Optional[bool] = typer.Option(False, help="Update ticket with AI results"),
    config_file: Optional[Path] = opt_config_file,  # type: ignore
    model: Optional[str] = opt_model,
):
    """
    Fetch and print a Jira ticket from the specified source.
    """

    console = init_logging([])

    # Validate source
    try:
        ticket_source = SourceFactory.create_source(
            source=source,
            config_file=config_file,
            ticket_url=ticket_url,
            ticket_username=ticket_username,
            ticket_api_key=ticket_api_key,
            ticket_id=ticket_id,
            model=model,
        )
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return

    try:
        issue_to_investigate = ticket_source.source.fetch_issue(id=ticket_id)  # type: ignore
        if issue_to_investigate is None:
            raise Exception(f"Issue {ticket_id} Not found")
    except Exception as e:
        logging.error(f"Failed to fetch issue from {source}", exc_info=e)
        console.print(
            f"[bold red]Error: Failed to fetch issue {ticket_id} from {source}.[/bold red]"
        )
        return

    with tool_result_storage() as tool_results_dir:
        ai = ticket_source.config.create_console_toolcalling_llm(model_name=model, tool_results_dir=tool_results_dir)

        # Render ticket-specific additions
        ticket_additions = load_and_render_prompt(
            prompt="builtin://_ticket_additions.jinja2",
            context={
                "source": source,
                "output_instructions": ticket_source.output_instructions,
            },
        )

        system_prompt = build_system_prompt(
            toolsets=ai.tool_executor.toolsets,
            runbooks=None,
            system_prompt_additions=ticket_additions,
            cluster_name=ticket_source.config.cluster_name,
            ask_user_enabled=False,
            prompt_component_overrides={},
        )
        console.print(
            f"[bold yellow]Analyzing ticket: {issue_to_investigate.name}...[/bold yellow]"
        )
        prompt = (
            prompt
            + f" for issue '{issue_to_investigate.name}' with description:'{issue_to_investigate.description}'"
        )

        ticket_user_prompt = generate_user_prompt(prompt, context={})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ticket_user_prompt},
        ]
        result = ai.call(messages)

        console.print(Rule())
        console.print(
            f"[bold green]AI analysis of {issue_to_investigate.url} {prompt}[/bold green]"
        )
        console.print(result.result.replace("\n", "\n\n"), style="bold green")  # type: ignore
        console.print(Rule())

        if update:
            ticket_source.source.write_back_result(issue_to_investigate.id, result)
            console.print(f"[bold]Updated ticket {issue_to_investigate.url}.[/bold]")
        else:
            console.print(
                f"[bold]Not updating ticket {issue_to_investigate.url}. Use the --update option to do so.[/bold]"
            )


@investigate_app.command()
def github(
    github_url: str = typer.Option(
        "https://api.github.com",
        help="The GitHub api base url (e.g: https://api.github.com)",
    ),
    github_owner: Optional[str] = typer.Option(
        None,
        help="The GitHub repository Owner, eg: if the repository url is https://github.com/HolmesGPT/holmesgpt, the owner is HolmesGPT",
    ),
    github_pat: str = typer.Option(
        None,
    ),
    github_repository: Optional[str] = typer.Option(
        None,
        help="The GitHub repository name, eg: if the repository url is https://github.com/HolmesGPT/holmesgpt, the repository name is holmesgpt",
    ),
    update: Optional[bool] = typer.Option(False, help="Update GitHub with AI results"),
    github_query: Optional[str] = typer.Option(
        "is:issue is:open",
        help="Investigate tickets matching a GitHub query (e.g. 'is:issue is:open')",
    ),
    # common options
    api_key: Optional[str] = opt_api_key,
    model: Optional[str] = opt_model,
    config_file: Optional[Path] = opt_config_file,  # type: ignore
    custom_toolsets: Optional[List[Path]] = opt_custom_toolsets,

    max_steps: Optional[int] = opt_max_steps,
    verbose: Optional[List[bool]] = opt_verbose,
):
    """
    Investigate a GitHub issue
    """
    console = init_logging(verbose)  # type: ignore

    config = Config.load_from_file(
        config_file,
        api_key=api_key,
        model=model,
        max_steps=max_steps,
        github_url=github_url,
        github_owner=github_owner,
        github_pat=github_pat,
        github_repository=github_repository,
        github_query=github_query,
        custom_toolsets_from_cli=custom_toolsets,
    )
    source = config.create_github_source()
    try:
        issues = source.fetch_issues()
    except Exception as e:
        logging.error("Failed to fetch issues from GitHub", exc_info=e)
        return

    console.print(
        f"[bold yellow]Analyzing {len(issues)} GitHub Issues.[/bold yellow] [red]Press Ctrl+C to stop.[/red]"
    )
    with tool_result_storage() as tool_results_dir:
        ai = config.create_console_toolcalling_llm(model_name=model, tool_results_dir=tool_results_dir)
        for i, issue in enumerate(issues):
            console.print(
                f"[bold yellow]Analyzing GitHub issue {i+1}/{len(issues)}: {issue.name}...[/bold yellow]"
            )

            result = _investigate_issue(ai, issue, config)

            console.print(Rule())
            console.print(f"[bold green]AI analysis of {issue.url}[/bold green]")
            console.print(Markdown(result.result.replace("\n", "\n\n")), style="bold green")  # type: ignore
            console.print(Rule())
            if update:
                source.write_back_result(issue.id, result)
                console.print(f"[bold]Updated ticket {issue.url}.[/bold]")
            else:
                console.print(
                    f"[bold]Not updating issue {issue.url}. Use the --update option to do so.[/bold]"
                )


@investigate_app.command()
def pagerduty(
    pagerduty_api_key: str = typer.Option(
        None,
        help="The PagerDuty API key.  This can be found in the PagerDuty UI under Integrations > API Access Keys.",
    ),
    pagerduty_user_email: Optional[str] = typer.Option(
        None,
        help="When --update is set, which user will be listed as the user who updated the ticket. (Must be the email of a valid user in your PagerDuty account.)",
    ),
    pagerduty_incident_key: Optional[str] = typer.Option(
        None,
        help="If provided, only analyze a single PagerDuty incident matching this key",
    ),
    update: Optional[bool] = typer.Option(
        False, help="Update PagerDuty with AI results"
    ),
    # common options
    api_key: Optional[str] = opt_api_key,
    model: Optional[str] = opt_model,
    config_file: Optional[Path] = opt_config_file,  # type: ignore
    custom_toolsets: Optional[List[Path]] = opt_custom_toolsets,

    max_steps: Optional[int] = opt_max_steps,
    verbose: Optional[List[bool]] = opt_verbose,
    json_output_file: Optional[str] = opt_json_output_file,
):
    """
    Investigate a PagerDuty incident
    """
    console = init_logging(verbose)

    config = Config.load_from_file(
        config_file,
        api_key=api_key,
        model=model,
        max_steps=max_steps,
        pagerduty_api_key=pagerduty_api_key,
        pagerduty_user_email=pagerduty_user_email,
        pagerduty_incident_key=pagerduty_incident_key,
        custom_toolsets_from_cli=custom_toolsets,
    )
    source = config.create_pagerduty_source()
    try:
        issues = source.fetch_issues()
    except Exception as e:
        logging.error("Failed to fetch issues from PagerDuty", exc_info=e)
        return

    console.print(
        f"[bold yellow]Analyzing {len(issues)} PagerDuty incidents.[/bold yellow] [red]Press Ctrl+C to stop.[/red]"
    )

    results = []
    with tool_result_storage() as tool_results_dir:
        ai = config.create_console_toolcalling_llm(model_name=model, tool_results_dir=tool_results_dir)
        for i, issue in enumerate(issues):
            console.print(
                f"[bold yellow]Analyzing PagerDuty incident {i+1}/{len(issues)}: {issue.name}...[/bold yellow]"
            )

            result = _investigate_issue(ai, issue, config)

            console.print(Rule())
            console.print(f"[bold green]AI analysis of {issue.url}[/bold green]")
            console.print(Markdown(result.result.replace("\n", "\n\n")), style="bold green")  # type: ignore
            console.print(Rule())
            if update:
                source.write_back_result(issue.id, result)
                console.print(f"[bold]Updated alert {issue.url}.[/bold]")
            else:
                console.print(
                    f"[bold]Not updating alert {issue.url}. Use the --update option to do so.[/bold]"
                )
            results.append({"issue": issue.model_dump(), "result": result.model_dump()})

    if json_output_file:
        write_json_file(json_output_file, results)


@investigate_app.command()
def opsgenie(
    opsgenie_api_key: str = typer.Option(None, help="The OpsGenie API key"),
    opsgenie_team_integration_key: str = typer.Option(
        None, help=OPSGENIE_TEAM_INTEGRATION_KEY_HELP
    ),
    opsgenie_query: Optional[str] = typer.Option(
        None,
        help="E.g. 'message: Foo' (see https://support.atlassian.com/opsgenie/docs/search-queries-for-alerts/)",
    ),
    update: Optional[bool] = typer.Option(
        False, help="Update OpsGenie with AI results"
    ),
    # common options
    api_key: Optional[str] = opt_api_key,
    model: Optional[str] = opt_model,
    config_file: Optional[Path] = opt_config_file,  # type: ignore
    custom_toolsets: Optional[List[Path]] = opt_custom_toolsets,

    max_steps: Optional[int] = opt_max_steps,
    verbose: Optional[List[bool]] = opt_verbose,
    documents: Optional[str] = opt_documents,
):
    """
    Investigate an OpsGenie alert
    """
    console = init_logging(verbose)  # type: ignore

    config = Config.load_from_file(
        config_file,
        api_key=api_key,
        model=model,
        max_steps=max_steps,
        opsgenie_api_key=opsgenie_api_key,
        opsgenie_team_integration_key=opsgenie_team_integration_key,
        opsgenie_query=opsgenie_query,
        custom_toolsets_from_cli=custom_toolsets,
    )
    source = config.create_opsgenie_source()
    try:
        issues = source.fetch_issues()
    except Exception as e:
        logging.error("Failed to fetch issues from OpsGenie", exc_info=e)
        return

    console.print(
        f"[bold yellow]Analyzing {len(issues)} OpsGenie alerts.[/bold yellow] [red]Press Ctrl+C to stop.[/red]"
    )
    with tool_result_storage() as tool_results_dir:
        ai = config.create_console_toolcalling_llm(model_name=model, tool_results_dir=tool_results_dir)
        for i, issue in enumerate(issues):
            console.print(
                f"[bold yellow]Analyzing OpsGenie alert {i+1}/{len(issues)}: {issue.name}...[/bold yellow]"
            )
            result = _investigate_issue(ai, issue, config)

            console.print(Rule())
            console.print(f"[bold green]AI analysis of {issue.url}[/bold green]")
            console.print(Markdown(result.result.replace("\n", "\n\n")), style="bold green")  # type: ignore
            console.print(Rule())
            if update:
                source.write_back_result(issue.id, result)
                console.print(f"[bold]Updated alert {issue.url}.[/bold]")
            else:
                console.print(
                    f"[bold]Not updating alert {issue.url}. Use the --update option to do so.[/bold]"
                )


@toolset_app.command("list")
def list_toolsets(
    verbose: Optional[List[bool]] = opt_verbose,
    config_file: Optional[Path] = opt_config_file,  # type: ignore
):
    """
    List build-in and custom toolsets status of CLI
    """
    console = init_logging(verbose)
    config = Config.load_from_file(config_file)
    cli_toolsets = config.toolset_manager.list_console_toolsets()

    pretty_print_toolset_status(cli_toolsets, console)


@toolset_app.command("refresh")
def refresh_toolsets(
    verbose: Optional[List[bool]] = opt_verbose,
    config_file: Optional[Path] = opt_config_file,  # type: ignore
):
    """
    Refresh build-in and custom toolsets status of CLI
    """
    console = init_logging(verbose)
    config = Config.load_from_file(config_file)
    cli_toolsets = config.toolset_manager.list_console_toolsets(refresh_status=True)
    pretty_print_toolset_status(cli_toolsets, console)


@toolset_app.command("config")
def config_toolset(
    verbose: Optional[List[bool]] = opt_verbose,
    config_file: Optional[Path] = opt_config_file,  # type: ignore
):
    """
    Interactive configuration editor for toolsets
    """
    console = init_logging(verbose)
    config = Config.load_from_file(config_file)
    run_toolset_config_tui(config, config_file, console)


@app.command()
def version() -> None:
    typer.echo(get_version())


def run():
    # Default to "ask" command when no subcommand is given
    if len(sys.argv) == 1:
        sys.argv.insert(1, "ask")
    app()


if __name__ == "__main__":
    run()
