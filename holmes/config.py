import logging
import os
import os.path
from enum import Enum
from pathlib import Path

display_logger = logging.getLogger("holmes.display.config")
from typing import TYPE_CHECKING, Any, List, Optional, Union

import sentry_sdk
import yaml  # type: ignore
from pydantic import (
    BaseModel,
    ConfigDict,
    FilePath,
    PrivateAttr,
    SecretStr,
)

from holmes.common.env_vars import ROBUSTA_CONFIG_PATH
from holmes.core.init_event import EventCallback, StatusEvent, StatusEventKind
from holmes.core.llm import DefaultLLM, LLMModelRegistry
from holmes.core.tools import Toolset
from holmes.core.tools_utils.tool_executor import ToolExecutor
from holmes.core.toolset_manager import ToolsetManager
from holmes.core.transformers.llm_summarize import LLMSummarizeTransformer
from holmes.plugins.runbooks import (
    RunbookCatalog,
    load_runbook_catalog,
)

# Source plugin imports moved to their respective create methods to speed up startup
if TYPE_CHECKING:
    from holmes.core.tool_calling_llm import ToolCallingLLM
    from holmes.plugins.destinations.slack import SlackDestination
    from holmes.plugins.sources.github import GitHubSource
    from holmes.plugins.sources.jira import JiraServiceManagementSource, JiraSource
    from holmes.plugins.sources.opsgenie import OpsGenieSource
    from holmes.plugins.sources.pagerduty import PagerDutySource
    from holmes.plugins.sources.prometheus.plugin import AlertManagerSource

from holmes.core.config import config_path_dir
from holmes.core.supabase_dal import SupabaseDal
from holmes.utils.definitions import RobustaConfig
from holmes.utils.pydantic_utils import RobustaBaseConfig, load_model_from_file

DEFAULT_CONFIG_LOCATION = os.path.join(config_path_dir, "config.yaml")


class SupportedTicketSources(str, Enum):
    JIRA_SERVICE_MANAGEMENT = "jira-service-management"
    PAGERDUTY = "pagerduty"


class Config(RobustaBaseConfig):
    model: Optional[str] = None
    _model_source: Optional[str] = None  # tracks where the model was set from
    api_key: Optional[SecretStr] = (
        None  # if None, read from OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT env var
    )
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    fast_model: Optional[str] = None
    max_steps: int = 100
    cluster_name: Optional[str] = None

    alertmanager_url: Optional[str] = None
    alertmanager_username: Optional[str] = None
    alertmanager_password: Optional[str] = None
    alertmanager_alertname: Optional[str] = None
    alertmanager_label: Optional[List[str]] = []
    alertmanager_file: Optional[FilePath] = None

    jira_url: Optional[str] = None
    jira_username: Optional[str] = None
    jira_api_key: Optional[SecretStr] = None
    jira_query: Optional[str] = ""

    github_url: Optional[str] = None
    github_owner: Optional[str] = None
    github_pat: Optional[SecretStr] = None
    github_repository: Optional[str] = None
    github_query: str = ""

    slack_token: Optional[SecretStr] = None
    slack_channel: Optional[str] = None

    pagerduty_api_key: Optional[SecretStr] = None
    pagerduty_user_email: Optional[str] = None
    pagerduty_incident_key: Optional[str] = None

    opsgenie_api_key: Optional[SecretStr] = None
    opsgenie_team_integration_key: Optional[SecretStr] = None
    opsgenie_query: Optional[str] = None

    custom_runbook_catalogs: List[Union[str, FilePath]] = []

    # custom_toolsets is passed from config file, and be used to override built-in toolsets, provides 'stable' customized toolset.
    # The status of custom toolsets can be cached.
    custom_toolsets: Optional[List[FilePath]] = None
    # custom_toolsets_from_cli is passed from CLI option `--custom-toolsets` as 'experimental' custom toolsets.
    # The status of toolset here won't be cached, so the toolset from cli will always be loaded when specified in the CLI.
    custom_toolsets_from_cli: Optional[List[FilePath]] = None
    # if True, we will try to load the Robusta AI model, in cli we aren't trying to load it.
    should_try_robusta_ai: bool = False

    toolsets: Optional[dict[str, dict[str, Any]]] = None
    mcp_servers: Optional[dict[str, dict[str, Any]]] = None
    additional_toolsets: Optional[List[Toolset]] = None

    _server_tool_executor: Optional[ToolExecutor] = None
    _agui_tool_executor: Optional[ToolExecutor] = None

    # TODO: Separate those fields to facade class, this shouldn't be part of the config.
    _toolset_manager: Optional[ToolsetManager] = PrivateAttr(None)
    _llm_model_registry: Optional[LLMModelRegistry] = PrivateAttr(None)
    _dal: Optional[SupabaseDal] = PrivateAttr(None)
    _config_file_path: Optional[Path] = PrivateAttr(None)

    @property
    def toolset_manager(self) -> ToolsetManager:
        if not self._toolset_manager:
            # Set the class-level default once before any transformers are
            # instantiated.  ToolsetManager no longer needs to know about it.
            if self.fast_model:
                LLMSummarizeTransformer.set_default_fast_model(self.fast_model)

            self._toolset_manager = ToolsetManager(
                toolsets=self.toolsets,
                mcp_servers=self.mcp_servers,
                custom_toolsets=self.custom_toolsets,
                custom_toolsets_from_cli=self.custom_toolsets_from_cli,
                custom_runbook_catalogs=self.custom_runbook_catalogs,
                config_file_path=self._config_file_path,
                additional_toolsets=self.additional_toolsets,
            )
        return self._toolset_manager

    @property
    def dal(self) -> SupabaseDal:
        if not self._dal:
            self._dal = SupabaseDal(self.cluster_name)  # type: ignore
        return self._dal

    @property
    def llm_model_registry(self) -> LLMModelRegistry:
        if not self._llm_model_registry:
            self._llm_model_registry = LLMModelRegistry(self, dal=self.dal)
        return self._llm_model_registry



    def log_useful_info(self):
        if self.llm_model_registry.models:
            display_logger.info(
                f"Loaded models: {list(self.llm_model_registry.models.keys())}"
            )
        else:
            display_logger.warning("No llm models were loaded")

    @classmethod
    def load_from_file(cls, config_file: Optional[Path], **kwargs) -> "Config":
        """
        Load configuration from file and merge with CLI options.

        Args:
            config_file: Path to configuration file
            **kwargs: CLI options to override config file values

        Returns:
            Config instance with merged settings
        """

        config_from_file: Optional[Config] = None
        if config_file is not None and config_file.exists():
            logging.debug(f"Loading config from {config_file}")
            config_from_file = load_model_from_file(cls, config_file)

        cli_options = {k: v for k, v in kwargs.items() if v is not None and v != []}

        if config_from_file is None:
            result = cls(**cli_options)
        else:
            logging.debug(f"Overriding config from cli options {cli_options}")
            merged_config = config_from_file.dict()
            merged_config.update(cli_options)
            result = cls(**merged_config)

        if config_file is not None and config_file.exists():
            result._config_file_path = config_file

        # Track where the model setting came from
        if "model" in cli_options:
            pass  # CLI --model flag: no source label needed (user just typed it)
        elif config_from_file is not None and config_from_file.model is not None:
            result._model_source = f"in {config_file}"
        # Fall through to env var check below

        if result.model is None:
            model_from_env = os.environ.get("MODEL")
            if model_from_env and model_from_env.strip():
                result.model = model_from_env
                result._model_source = "via $MODEL"

        result.log_useful_info()
        return result

    @classmethod
    def load_from_env(cls):
        kwargs = {}
        for field_name in [
            "model",
            "fast_model",
            "api_key",
            "api_base",
            "api_version",
            "max_steps",
            "alertmanager_url",
            "alertmanager_username",
            "alertmanager_password",
            "jira_url",
            "jira_username",
            "jira_api_key",
            "jira_query",
            "slack_token",
            "slack_channel",
            "github_url",
            "github_owner",
            "github_repository",
            "github_pat",
            "github_query",
        ]:
            val = os.getenv(field_name.upper(), None)
            if val is not None:
                kwargs[field_name] = val
        kwargs["cluster_name"] = Config.__get_cluster_name()
        kwargs["should_try_robusta_ai"] = True
        result = cls(**kwargs)
        if "model" in kwargs:
            result._model_source = "via $MODEL"
        result.log_useful_info()
        return result

    @staticmethod
    def __get_cluster_name() -> Optional[str]:
        config_file_path = ROBUSTA_CONFIG_PATH
        env_cluster_name = os.environ.get("CLUSTER_NAME")
        if env_cluster_name:
            return env_cluster_name

        if not os.path.exists(config_file_path):
            logging.info(f"No robusta config in {config_file_path}")
            return None

        logging.info(f"loading config {config_file_path}")
        with open(config_file_path) as file:
            yaml_content = yaml.safe_load(file)
            config = RobustaConfig(**yaml_content)
            return config.global_config.get("cluster_name")

        return None

    def get_runbook_catalog(self) -> Optional[RunbookCatalog]:
        runbook_catalog = load_runbook_catalog(
            dal=self.dal, custom_catalog_paths=self.custom_runbook_catalogs
        )
        return runbook_catalog

    def create_console_tool_executor(
        self,
        # 可选的数据访问层；某些 toolset 需要通过它访问数据库或平台配置。
        dal: Optional["SupabaseDal"],
        # 是否强制刷新 toolset 状态，例如重新探测某些集成当前是否可用。
        refresh_status: bool = False,
        # 事件回调，用于把 toolset 加载进度或状态变化通知给 CLI/UI。
        on_event: EventCallback = None,
    ) -> ToolExecutor:
        """
        Creates a ToolExecutor instance configured for CLI usage. This executor manages the available tools
        and their execution in the command-line interface.

        The method loads toolsets in this order, with later sources overriding earlier ones:
        1. Built-in toolsets (tagged as CORE or CLI)
        2. toolsets from config file will override and be merged into built-in toolsets with the same name.
        3. Custom toolsets from config files which can not override built-in toolsets
        """
        # 先让 toolset_manager 解析并筛出“适用于 console”的工具集合。
        # 这里同时会按既定优先级完成内置 toolset、配置文件覆盖和自定义 toolset 的合并。
        cli_toolsets = self.toolset_manager.list_console_toolsets(
            dal=dal, refresh_status=refresh_status, on_event=on_event
        )
        # 再把最终的 toolset 列表封装成 ToolExecutor。
        # 原因是上层不需要关心每个工具来自哪个 toolset，只需要一个统一的执行入口。
        return ToolExecutor(cli_toolsets, on_event=on_event)

    def create_agui_tool_executor(self, dal: Optional["SupabaseDal"]) -> ToolExecutor:
        """
        Creates ToolExecutor for the AG-UI server endpoints
        """

        if self._agui_tool_executor:
            return self._agui_tool_executor

        # Use same toolset as CLI for AG-UI front-end.
        agui_toolsets = self.toolset_manager.list_console_toolsets(
            dal=dal, refresh_status=True
        )

        self._agui_tool_executor = ToolExecutor(agui_toolsets)

        return self._agui_tool_executor

    def create_tool_executor(self, dal: Optional["SupabaseDal"]) -> ToolExecutor:
        """
        Creates ToolExecutor for the server endpoints
        """

        if self._server_tool_executor:
            return self._server_tool_executor

        toolsets = self.toolset_manager.list_server_toolsets(dal=dal)

        self._server_tool_executor = ToolExecutor(toolsets)

        logging.debug(
            f"Starting AI session with tools: {[tn for tn in self._server_tool_executor.tools_by_name.keys()]}"
        )

        return self._server_tool_executor

    def refresh_server_tool_executor(
        self, dal: Optional["SupabaseDal"]
    ) -> list[tuple[str, str, str]]:
        if not self._server_tool_executor:
            self.create_tool_executor(dal)
            return []

        current_toolsets = self._server_tool_executor.toolsets
        new_toolsets, changes = (
            self.toolset_manager.refresh_server_toolsets_and_get_changes(
                current_toolsets, dal
            )
        )

        if changes:
            self._server_tool_executor = ToolExecutor(new_toolsets)

        return [(name, old.value, new.value) for name, old, new in changes]

    def create_console_toolcalling_llm(
        self,
        # 可选的数据访问层，某些 toolset 会通过它读写持久化数据。
        dal: Optional["SupabaseDal"] = None,
        # 是否强制刷新 console toolset 的状态缓存。
        refresh_toolsets: bool = False,
        # tracing 对象，用于记录模型与工具调用链路。
        tracer=None,
        # 可覆盖配置中的默认模型名。
        model_name: Optional[str] = None,
        # 工具结果落盘目录；传入后可保存工具执行产物供后续查看。
        tool_results_dir: Optional[Path] = None,
        # 初始化和执行过程中的事件回调，通常用于 CLI/交互界面刷新进度。
        on_event: EventCallback = None,
    ) -> "ToolCallingLLM":
        # 延迟导入以避免模块初始化时产生不必要的循环依赖或启动开销。
        from holmes.core.tool_calling_llm import ToolCallingLLM

        # 先创建 LLM，再创建 tool executor。
        # 原因是 console 场景下 toolset 加载过程会展示初始化进度，
        # 先拿到 LLM 信息可以更早把模型名显示给用户，提升可观测性。
        llm = self._get_llm(tracer=tracer, model_key=model_name, on_event=on_event)
        # 创建面向 console 的工具执行器，它负责装配可用工具及其状态。
        tool_executor = self.create_console_tool_executor(dal, refresh_toolsets, on_event=on_event)
        # 把“模型 + 工具执行器 + 最大步骤数”组合成统一的 ToolCallingLLM。
        # 这样上层只需要面向一个对象调用，不必分别协调模型和工具系统。
        return ToolCallingLLM(
            tool_executor,
            self.max_steps,
            llm,
            tool_results_dir=tool_results_dir,
        )

    def create_agui_toolcalling_llm(
        self,
        dal: Optional["SupabaseDal"] = None,
        model: Optional[str] = None,
        tracer=None,
        tool_results_dir: Optional[Path] = None,
    ) -> "ToolCallingLLM":
        tool_executor = self.create_agui_tool_executor(dal)
        from holmes.core.tool_calling_llm import ToolCallingLLM

        return ToolCallingLLM(
            tool_executor,
            self.max_steps,
            self._get_llm(model, tracer),
            tool_results_dir=tool_results_dir,
        )

    def create_toolcalling_llm(
        self,
        dal: Optional["SupabaseDal"] = None,
        model: Optional[str] = None,
        tracer=None,
        tool_results_dir: Optional[Path] = None,
    ) -> "ToolCallingLLM":
        tool_executor = self.create_tool_executor(dal)
        from holmes.core.tool_calling_llm import ToolCallingLLM

        return ToolCallingLLM(
            tool_executor,
            self.max_steps,
            self._get_llm(model, tracer),
            tool_results_dir=tool_results_dir,
        )

    def validate_jira_config(self):
        if self.jira_url is None:
            raise ValueError("--jira-url must be specified")
        if not (
            self.jira_url.startswith("http://") or self.jira_url.startswith("https://")
        ):
            raise ValueError("--jira-url must start with http:// or https://")
        if self.jira_username is None:
            raise ValueError("--jira-username must be specified")
        if self.jira_api_key is None:
            raise ValueError("--jira-api-key must be specified")

    def create_jira_source(self) -> "JiraSource":
        from holmes.plugins.sources.jira import JiraSource

        self.validate_jira_config()

        return JiraSource(
            url=self.jira_url,  # type: ignore
            username=self.jira_username,  # type: ignore
            api_key=self.jira_api_key.get_secret_value(),  # type: ignore
            jql_query=self.jira_query,  # type: ignore
        )

    def create_jira_service_management_source(self) -> "JiraServiceManagementSource":
        from holmes.plugins.sources.jira import JiraServiceManagementSource

        self.validate_jira_config()

        return JiraServiceManagementSource(
            url=self.jira_url,  # type: ignore
            username=self.jira_username,  # type: ignore
            api_key=self.jira_api_key.get_secret_value(),  # type: ignore
            jql_query=self.jira_query,  # type: ignore
        )

    def create_github_source(self) -> "GitHubSource":
        from holmes.plugins.sources.github import GitHubSource

        if not self.github_url or not (
            self.github_url.startswith("http://")
            or self.github_url.startswith("https://")
        ):
            raise ValueError("--github-url must start with http:// or https://")
        if self.github_owner is None:
            raise ValueError("--github-owner must be specified")
        if self.github_repository is None:
            raise ValueError("--github-repository must be specified")
        if self.github_pat is None:
            raise ValueError("--github-pat must be specified")

        return GitHubSource(
            url=self.github_url,
            owner=self.github_owner,
            pat=self.github_pat.get_secret_value(),
            repository=self.github_repository,
            query=self.github_query,
        )

    def create_pagerduty_source(self) -> "PagerDutySource":
        from holmes.plugins.sources.pagerduty import PagerDutySource

        if self.pagerduty_api_key is None:
            raise ValueError("--pagerduty-api-key must be specified")

        return PagerDutySource(
            api_key=self.pagerduty_api_key.get_secret_value(),
            user_email=self.pagerduty_user_email,  # type: ignore
            incident_key=self.pagerduty_incident_key,
        )

    def create_opsgenie_source(self) -> "OpsGenieSource":
        from holmes.plugins.sources.opsgenie import OpsGenieSource

        if self.opsgenie_api_key is None:
            raise ValueError("--opsgenie-api-key must be specified")

        return OpsGenieSource(
            api_key=self.opsgenie_api_key.get_secret_value(),
            query=self.opsgenie_query,  # type: ignore
            team_integration_key=(
                self.opsgenie_team_integration_key.get_secret_value()
                if self.opsgenie_team_integration_key
                else None
            ),
        )

    def create_alertmanager_source(self) -> "AlertManagerSource":
        from holmes.plugins.sources.prometheus.plugin import AlertManagerSource

        return AlertManagerSource(
            url=self.alertmanager_url,  # type: ignore
            username=self.alertmanager_username,
            alertname_filter=self.alertmanager_alertname,  # type: ignore
            label_filter=self.alertmanager_label,  # type: ignore
            filepath=self.alertmanager_file,
        )

    def create_slack_destination(self) -> "SlackDestination":
        from holmes.plugins.destinations.slack import SlackDestination

        if self.slack_token is None:
            raise ValueError("--slack-token must be specified")
        if self.slack_channel is None:
            raise ValueError("--slack-channel must be specified")
        return SlackDestination(self.slack_token.get_secret_value(), self.slack_channel)

    @staticmethod
    def _format_token_count(n: int) -> str:
        """Format a token count for display: 1048576 → '1M', 32768 → '32K'."""
        if n >= 1_000_000:
            value = n / 1_000_000
            return f"{int(value)}M" if value == int(value) else f"{value:.1f}M"
        if n >= 1_000:
            value = n / 1_000
            return f"{int(value)}K" if value == int(value) else f"{value:.0f}K"
        return str(n)

    # TODO: move this to the llm model registry
    def _get_llm(
        self,
        # 可选的模型键；传入后优先选择该模型，否则走配置里的默认模型。
        model_key: Optional[str] = None,
        # tracing 对象，用于记录模型调用链路。
        tracer=None,
        # 事件回调，通常用于把“模型已加载”等状态通知给 CLI/UI。
        on_event: EventCallback = None,
    ) -> "DefaultLLM":
        # 把用户请求的模型记到 Sentry 上，方便线上排查“实际请求了哪个模型”。
        sentry_sdk.set_tag("requested_model", model_key)
        # 从模型注册表中取出最终生效的模型配置。
        model_entry = self.llm_model_registry.get_model_params(model_key)
        # 转成普通 dict，并去掉值为 None 的字段，避免把空配置继续传下去。
        model_params = model_entry.model_dump(exclude_none=True)
        # 先拿全局默认的 api_base / api_version，后面如果模型级别有覆盖再替换。
        api_base = self.api_base
        api_version = self.api_version
        # 某些模型走 Robusta 自己的凭证体系，因此需要单独识别。
        is_robusta_model = model_params.pop("is_robusta_model", False)
        sentry_sdk.set_tag("is_robusta_model", is_robusta_model)
        if is_robusta_model:
            # Robusta 模型的 token 可能会过期并被动态刷新，所以不能在模型注册阶段就固化。
            # 因此这里在真正创建 LLM 前临时取一次最新凭证。
            account_id, token = self.dal.get_ai_credentials()
            api_key = f"{account_id} {token}"
        else:
            # 非 Robusta 模型直接从模型配置里读取 api_key。
            api_key = model_params.pop("api_key", None)
            if api_key is not None:
                # 配置层通常把密钥包装成 Secret 类型，这里要还原成真实字符串给底层客户端。
                api_key = api_key.get_secret_value()

        # `model` 是真正传给底层 LLM SDK/provider 的模型标识。
        model = model_params.pop("model")
        # It's ok if the model does not have api base and api version, which are defaults to None.
        # Handle both api_base and base_url - api_base takes precedence
        # 兼容两种命名：`api_base` 和 `base_url`，并且优先使用 `api_base`。
        model_api_base = model_params.pop("api_base", None)
        model_base_url = model_params.pop("base_url", None)
        # 优先级是：模型级配置 > 全局配置。
        api_base = model_api_base or model_base_url or api_base
        # api_version 也允许被模型级配置覆盖。
        api_version = model_params.pop("api_version", api_version)
        # 展示名优先取显式 name，其次取 model_key，最后退回到底层 model 标识。
        # 这样做的原因是展示名不一定等于 provider 真正使用的模型 ID，分开更利于用户理解和排查。
        model_name = model_params.pop("name", None) or model_key or model
        sentry_sdk.set_tag("model_name", model_name)
        # 统一在这里构造 DefaultLLM，把通用参数和剩余 provider 参数一起传入。
        llm = DefaultLLM(
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            args=model_params,
            tracer=tracer,
            name=model_name,
            is_robusta_model=is_robusta_model,
        )  # type: ignore
        # 读取模型能力信息并格式化，主要用于初始化日志和界面提示。
        context_size = self._format_token_count(llm.get_context_window_size())
        max_response = self._format_token_count(llm.get_maximum_output_token())
        # 告诉用户当前模型来源于哪里：默认值还是显式配置。
        if self._model_source and self._model_source != "default":
            source_hint = f"configured {self._model_source}"
        else:
            source_hint = "default, change with --model, for all options see https://holmesgpt.dev/ai-providers"
        # 组装一条可读性较高的模型加载提示，展示模型名、上下文窗口和最大输出长度。
        msg = f"Model: {model_name}, {context_size} context, {max_response} max response ({source_hint})"
        display_logger.info(msg)
        if on_event is not None:
            # 如果有 UI/CLI 监听事件，就把“模型已加载”事件发出去，驱动界面更新。
            on_event(StatusEvent(kind=StatusEventKind.MODEL_LOADED, name=model_name, message=msg))
        # 返回已经准备好的 LLM 实例，供上层继续装配到 ToolCallingLLM 中。
        return llm

    def get_models_list(self) -> List[str]:
        if self.llm_model_registry and self.llm_model_registry.models:
            return list(self.llm_model_registry.models.keys())

        return []


class TicketSource(BaseModel):
    config: Config
    output_instructions: list[str]
    source: Union["JiraServiceManagementSource", "PagerDutySource"]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SourceFactory(BaseModel):
    @staticmethod
    def create_source(
        source: SupportedTicketSources,
        config_file: Optional[Path],
        ticket_url: Optional[str],
        ticket_username: Optional[str],
        ticket_api_key: Optional[str],
        ticket_id: Optional[str],
        model: Optional[str] = None,
    ) -> TicketSource:
        from holmes.plugins.sources.jira import JiraServiceManagementSource
        from holmes.plugins.sources.pagerduty import PagerDutySource

        TicketSource.model_rebuild()
        supported_sources = [s.value for s in SupportedTicketSources]
        if source not in supported_sources:
            raise ValueError(
                f"Source '{source}' is not supported. Supported sources: {', '.join(supported_sources)}"
            )

        if source == SupportedTicketSources.JIRA_SERVICE_MANAGEMENT:
            config = Config.load_from_file(
                config_file=config_file,
                api_key=None,
                model=model,
                max_steps=None,
                jira_url=ticket_url,
                jira_username=ticket_username,
                jira_api_key=ticket_api_key,
                jira_query=None,
                custom_toolsets=None,
            )

            if not (
                config.jira_url
                and config.jira_username
                and config.jira_api_key
                and ticket_id
            ):
                raise ValueError(
                    "URL, username, API key, and ticket ID are required for jira-service-management"
                )

            output_instructions = [
                "All output links/urls must **always** be of this format : [link text here|http://your.url.here.com] and **never*** the format [link text here](http://your.url.here.com)"
            ]
            source_instance = config.create_jira_service_management_source()
            return TicketSource(
                config=config,
                output_instructions=output_instructions,
                source=source_instance,
            )

        elif source == SupportedTicketSources.PAGERDUTY:
            config = Config.load_from_file(
                config_file=config_file,
                api_key=None,
                model=model,
                max_steps=None,
                pagerduty_api_key=ticket_api_key,
                pagerduty_user_email=ticket_username,
                pagerduty_incident_key=None,
                custom_toolsets=None,
            )

            if not (
                config.pagerduty_user_email and config.pagerduty_api_key and ticket_id
            ):
                raise ValueError(
                    "username, API key, and ticket ID are required for pagerduty"
                )

            output_instructions = [
                "All output links/urls must **always** be of this format : \n link text here: http://your.url.here.com\n **never*** use the url the format [link text here](http://your.url.here.com)"
            ]
            source_instance = config.create_pagerduty_source()  # type: ignore
            return TicketSource(
                config=config,
                output_instructions=output_instructions,
                source=source_instance,
            )

        else:
            raise NotImplementedError(f"Source '{source}' is not yet implemented")
