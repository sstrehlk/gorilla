import json
import os
import re
import time
from typing import Any, Optional

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.openvino_fc_support.parser_registry import (
    parse_openvino_fc_response,
)
from bfcl_eval.model_handler.local_inference.openvino_fc_support.template_registry import (
    apply_openvino_fc_chat_template,
)
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import contain_multi_turn_interaction
from overrides import EnforceOverrides, override


class BaseOpenVINOHandler(BaseHandler, EnforceOverrides):
    """
    Base handler for running OpenVINO IR models in-process (without a separate server).

    Unlike OSSHandler (which launches a vLLM/SGLang server), this handler loads the
    model directly into the current process and runs inference locally. This makes it
    suitable for OpenVINO IR format models used with either:
      - optimum-intel  (OVModelForCausalLM, see OpenVINOOptimumHandler)
      - openvino-genai (LLMPipeline, see OpenVINOGenAIHandler)

    Subclasses must implement:
      - _load_model(model_path, device)  – load the backend-specific model
      - _generate(formatted_prompt, max_new_tokens) -> str  – run inference
      - _format_prompt(messages, function) -> str  – (optional) build the prompt string

    The default _format_prompt uses the tokenizer's apply_chat_template, which works
    for most instruction-tuned models. Override it for custom formatting.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name
        self.model_style = ModelStyle.OSSMODEL
        self.tokenizer = None
        self.max_context_length: Optional[int] = None

    # ------------------------------------------------------------------
    # Inference entry-point (prompting only, no FC server mode)
    # ------------------------------------------------------------------

    @override
    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        return super().inference(test_entry, include_input_log, exclude_state_log)

    # ------------------------------------------------------------------
    # Model lifecycle (called by _llm_response_generation.py)
    # ------------------------------------------------------------------

    def load_model(
        self,
        local_model_path: Optional[str] = None,
        openvino_device: str = "CPU",
        ov_config: Optional[str] = None,
    ) -> None:
        """
        Load the tokenizer and the backend-specific model into memory.

        Args:
            local_model_path: Path to a local directory that contains the OpenVINO IR
                model files (.xml/.bin) together with the tokenizer config.  If None,
                self.model_name_huggingface is used (works when the model is cached in
                HF_HOME or when the backend supports HF hub download, e.g. optimum-intel).
            openvino_device: OpenVINO compute device – "CPU", "GPU", or "NPU".
            ov_config: Optional path to a JSON file (or a JSON string) with OpenVINO device
                properties, in the same shape used by the pnp-validation testplan configs:
                ``{"config": {"DEVICE_PROPERTIES": {"<DEVICE>": {...}}}}``. The properties
                matching ``openvino_device`` (or its device family, e.g. "GPU" for "GPU.0")
                are extracted and passed to ``_load_model`` as ``device_properties``.
        """
        from transformers import AutoConfig, AutoTokenizer

        model_path = (
            local_model_path if local_model_path is not None else self.model_name_huggingface
        )

        load_kwargs: dict = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True,
        }
        if local_model_path is not None:
            load_kwargs["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
        config = AutoConfig.from_pretrained(**load_kwargs)
        self.model_path = model_path
        apply_openvino_fc_chat_template(self.tokenizer, model_path)

        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        elif (
            self.tokenizer.model_max_length is not None
            and self.tokenizer.model_max_length < 1_000_000
        ):
            self.max_context_length = self.tokenizer.model_max_length
        else:
            self.max_context_length = 4096  # safe fallback

        print(f"Max context length: {self.max_context_length}")

        device_properties = self._parse_ov_config(ov_config, openvino_device)
        if device_properties:
            print(f"Applying OpenVINO device properties for {openvino_device}: {device_properties}")

        self._load_model(model_path, device=openvino_device, device_properties=device_properties)
        print(f"OpenVINO model loaded on device: {openvino_device}")

    @staticmethod
    def _parse_ov_config(ov_config: Optional[str], device: str) -> dict:
        """Parse an OpenVINO config file/JSON string and return the flat device-property
        dict applicable to ``device``.

        Expected shape (same as the pnp-validation testplan ``testplan_ov_config.json``):
        ``{"config": {"DEVICE_PROPERTIES": {"GPU": {"ATTENTION_BACKEND": "SDPA"}}}}``.
        The device family (e.g. "GPU" for a device string of "GPU.0") is used as a
        fallback lookup key if an exact match (e.g. "GPU.0") is not present. A flat dict
        with no "DEVICE_PROPERTIES" wrapper is also accepted and applied as-is. Returns
        ``{}`` if ``ov_config`` is empty or nothing matches.
        """
        if not ov_config:
            return {}

        if os.path.isfile(ov_config):
            with open(ov_config, "r", encoding="utf-8") as ov_config_file:
                raw = json.load(ov_config_file)
        else:
            raw = json.loads(ov_config)

        config = raw.get("config", raw) if isinstance(raw, dict) else {}
        if not isinstance(config, dict):
            return {}

        device_properties = config.get("DEVICE_PROPERTIES")
        if isinstance(device_properties, dict):
            device_key = device.upper()
            if device_key in device_properties:
                return dict(device_properties[device_key])
            device_family = device_key.split(".")[0]
            return dict(device_properties.get(device_family, {}))

        return {k: v for k, v in config.items() if k != "DEVICE_PROPERTIES"}

    def shutdown_local_server(self) -> None:
        """Release model resources (mirrors the OSSHandler interface)."""
        self._unload_model()
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    def _load_model(
        self, model_path: str, device: str = "CPU", device_properties: Optional[dict] = None
    ) -> None:
        """Load the backend-specific model.  Must be implemented by subclasses.

        ``device_properties`` (parsed from an optional ``--ov-config`` file/string, see
        ``load_model``/``_parse_ov_config``) contains OpenVINO plugin properties
        (e.g. ``ATTENTION_BACKEND``, ``INFERENCE_PRECISION_HINT``, ``KV_CACHE_PRECISION``)
        that subclasses should apply when constructing their pipeline/model.
        """
        raise NotImplementedError

    def _unload_model(self) -> None:
        """Release backend-specific resources.  May be overridden by subclasses."""
        # Default: nothing to do. Subclasses that hold GPU/device memory should
        # set their model reference to None here.
        return

    def _generate(self, formatted_prompt: str, max_new_tokens: int) -> str:
        """
        Run text generation and return only the *newly generated* text
        (i.e. not including the prompt itself).  Must be implemented by subclasses.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Prompt formatting (can be overridden per model / per format)
    # ------------------------------------------------------------------

    def _format_prompt(self, messages: list[dict], function: list[dict]) -> str:  # noqa: ARG002
        """
        Build the full prompt string from the messages list.

        The default implementation uses the tokenizer's built-in chat template
        (identical to QuickTestingOSSHandler).  Override this method in subclasses
        that need custom prompt construction.  The ``function`` argument is available
        for subclasses that embed function docs directly into the prompt.

        In prompting mode the assistant response is plain text (no structured
        ``tool_calls``), so some chat templates (e.g. Mistral) raise a
        TemplateError when they see a ``tool`` role message that is not preceded
        by an assistant message with ``tool_calls``.  To avoid this we convert
        ``tool`` role messages to ``user`` role messages before rendering.
        """
        sanitized_messages = []
        for msg in messages:
            if msg.get("role") == "tool":
                sanitized_messages.append(
                    {"role": "user", "content": f"[TOOL RESULT] {msg['content']}"}
                )
            else:
                sanitized_messages.append(msg)
        return self.tokenizer.apply_chat_template(
            sanitized_messages, add_generation_prompt=True, tokenize=False
        )

    # ------------------------------------------------------------------
    # Decoding helpers (prompting mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove <think>...</think> blocks produced by reasoning models (e.g. Qwen3).
        Also handles unclosed <think> blocks (when generation was cut off mid-thought).
        Also strips gpt-oss-20b style 'analysis...assistantfinal<answer>' preamble.
        """
        # Remove complete <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove unclosed <think> block (truncated generation)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        # Gemma-4 wraps its response in a "channel" delimited by dedicated
        # start/end token IDs (see OVMS's Gemma4ReasoningParser). Those tokens
        # are silently dropped during detokenization, so the only visible
        # leftover is a literal "thought\n" line-prefix (sometimes doubled)
        # near the very start of the text. Tolerate a small amount of leading
        # garbage (e.g. corrupted/garbled tokens produced by GPU-side
        # degeneration) before the marker, bounded to 20 characters so we
        # don't accidentally strip legitimate content that happens to contain
        # this substring further into the response.
        # EXPERIMENTAL: broadened from a strict `^` anchor - see
        # agents/pending_bfcl_fixes_proposals.md section 1 for the rationale
        # and the risk trade-off (approximates OVMS's token-boundary-based
        # Gemma4ReasoningParser via a text heuristic; not a faithful port).
        text = re.sub(r"^.{0,20}?(?:thought\n)+", "", text, count=1, flags=re.DOTALL)
        # gpt-oss-20b outputs chain-of-thought followed by "assistantfinal<answer>"
        # or "assistantcommentary to=functions...". Extract only the final answer.
        # For assistantfinal: use rfind to get the definitive last output.
        # For assistantcommentary: use find (first occurrence) so that all
        # subsequent 'assistantcommentary to=functions...' blocks are retained
        # for the regex to extract (handles multi-call steps like
        # 'assistantcommentary to=X json{...}assistantanalysis...assistantcommentary to=Y json{...}').
        idx = text.rfind("assistantfinal")
        if idx != -1:
            text = text[idx + len("assistantfinal"):]
        else:
            idx = text.find("assistantcommentary")
            if idx != -1:
                text = text[idx + len("assistantcommentary"):]
            else:
                # Fallback: bare 'final*' markers without 'assistant' prefix
                for marker in ("finalanalysis", "finalcommentary"):
                    idx = text.rfind(marker)
                    if idx != -1:
                        text = text[idx + len(marker):]
                        break
        return text.strip()

    @staticmethod
    def _extract_function_calls(text: str) -> str:
        """Convert model output in 'to=functions.X ...' format to Python-style
        function call(s) expected by the AST decoder.

        Handles (with and without the 'functions.' prefix):
          - 'to=functions.X json{...}'        (space + json keyword)
          - 'to=functions.X commentary{...}'  (space + commentary keyword)
          - 'to=functions.Xjson{...}'         (json directly attached to name)
          - 'to=functions.Xcommentary{...}'   (commentary directly attached)
          - 'to=X json{...}'                  (no functions. prefix, space + json)
          - 'to=Xjson{...}'                   (no functions. prefix, directly attached)
          - 'to=X {}'                          (bare space + braces, no keyword)
          - JSON bodies with one level of nested braces: {"key": {"nested": "val"}}

        Falls through unchanged if the pattern is not found.
        """
        # JSON body: handles one level of nested braces (e.g. {"parameters":{}})
        _JSON = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        pattern = re.compile(
            r"to=(?:functions\.)?([\.\w]+?)(?:json|commentary|\s+(?:json|commentary)|\s*(?=\{))("
            + _JSON + r")",
            re.DOTALL,
        )
        matches = pattern.findall(text)
        if not matches:
            return text

        calls = []
        for func_name, json_str in matches:
            try:
                args = json.loads(json_str)
                args_str = ", ".join(
                    f"{k}={repr(v)}" for k, v in args.items()
                )
                calls.append(f"{func_name}({args_str})")
            except json.JSONDecodeError:
                return text  # fall through to default parsing
        return ", ".join(calls)

    def _preprocess_result(self, result: str) -> str:
        text = self._strip_thinking_tags(result)
        return self._extract_function_calls(text)

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        if self.is_fc_model and isinstance(result, list):
            return result
        return default_decode_ast_prompting(
            self._preprocess_result(result), language, has_tool_call_tag
        )

    @override
    def decode_execute(self, result, has_tool_call_tag):
        if self.is_fc_model and isinstance(result, list):
            return convert_to_function_call(result)
        return default_decode_execute_prompting(
            self._preprocess_result(result), has_tool_call_tag
        )

    # ------------------------------------------------------------------
    # Prompting pipeline methods (mirrors OSSHandler)
    # ------------------------------------------------------------------

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        return {"message": [], "function": functions}

    @override
    def _query_prompting(self, inference_data: dict):
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))

        if self.max_context_length < input_token_count + 2:
            # Prompt already exceeds context window; request a minimal budget
            max_new_tokens = 1000
        else:
            # Cap at 2048 tokens to match OVMS's own BFCL test setup, which pins
            # max_completion_tokens=2048 for every request (see OpenAICompletionsHandler
            # in model_server/demos/continuous_batching/accuracy/gorilla.patch). A much
            # larger budget (previously 16384) let occasional degenerate/repetitive
            # generations run for a very long time before being cut off.
            max_new_tokens = min(
                2048,
                self.max_context_length - input_token_count - 2,
            )

        start_time = time.time()
        generated_text = self._generate(formatted_prompt, max_new_tokens)
        latency = time.time() - start_time

        output_token_count = len(self.tokenizer.tokenize(generated_text))

        return (
            {
                "text": generated_text,
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
            },
            latency,
        )

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return {
            "model_responses": api_response["text"],
            "input_token": api_response["input_tokens"],
            "output_token": api_response["output_tokens"],
        }

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )
        return inference_data

    # ------------------------------------------------------------------
    # FC (Function Calling) pipeline methods
    # ------------------------------------------------------------------

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        tools = [
            tool if "function" in tool else {"type": "function", "function": tool}
            for tool in tools
        ]
        inference_data["tools"] = tools
        return inference_data

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools: list[dict] = inference_data.get("tools", [])
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        try:
            formatted_prompt: str = self.tokenizer.apply_chat_template(
                message,
                tools=tools if tools else None,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except Exception:
            # Fallback: template does not support tools – render without
            formatted_prompt = self.tokenizer.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False
            )

        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
        if self.max_context_length < input_token_count + 2:
            max_new_tokens = 1000
        else:
            # Cap at 2048 tokens to match OVMS's own BFCL test setup (see
            # OpenAICompletionsHandler.max_completion_tokens in
            # model_server/demos/continuous_batching/accuracy/gorilla.patch).
            max_new_tokens = min(2048, self.max_context_length - input_token_count - 2)

        start_time = time.time()
        generated_text = self._generate(formatted_prompt, max_new_tokens)
        latency = time.time() - start_time

        output_token_count = len(self.tokenizer.tokenize(generated_text))

        return (
            {
                "text": generated_text,
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "tools": tools,
            },
            latency,
        )

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        text: str = api_response["text"]
        # Strip thinking tags (e.g. Qwen3 reasoning models)
        text = self._strip_thinking_tags(text)

        parsed = parse_openvino_fc_response(
            text,
            model_name=self.model_name_huggingface,
            model_path=getattr(self, "model_path", ""),
            tools=api_response.get("tools"),
        )

        return {
            "model_responses": parsed["model_responses"],
            "model_responses_message_for_chat_history": parsed[
                "model_responses_message_for_chat_history"
            ],
            "tool_call_ids": parsed["tool_call_ids"],
            "input_token": api_response["input_tokens"],
            "output_token": api_response["output_tokens"],
        }

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "content": execution_result,
                    "tool_call_id": tool_call_id,
                }
            )
        return inference_data
