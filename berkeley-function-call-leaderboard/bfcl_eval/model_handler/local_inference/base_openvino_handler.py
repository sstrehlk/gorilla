import json
import re
import time
from typing import Any, Optional

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
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
        if contain_multi_turn_interaction(test_entry["id"]):
            return self.inference_multi_turn_prompting(
                test_entry, include_input_log, exclude_state_log
            )
        else:
            return self.inference_single_turn_prompting(test_entry, include_input_log)

    # ------------------------------------------------------------------
    # Model lifecycle (called by _llm_response_generation.py)
    # ------------------------------------------------------------------

    def load_model(
        self,
        local_model_path: Optional[str] = None,
        openvino_device: str = "CPU",
    ) -> None:
        """
        Load the tokenizer and the backend-specific model into memory.

        Args:
            local_model_path: Path to a local directory that contains the OpenVINO IR
                model files (.xml/.bin) together with the tokenizer config.  If None,
                self.model_name_huggingface is used (works when the model is cached in
                HF_HOME or when the backend supports HF hub download, e.g. optimum-intel).
            openvino_device: OpenVINO compute device – "CPU", "GPU", or "NPU".
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

        self._load_model(model_path, device=openvino_device)
        print(f"OpenVINO model loaded on device: {openvino_device}")

    def shutdown_local_server(self) -> None:
        """Release model resources (mirrors the OSSHandler interface)."""
        self._unload_model()
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, device: str = "CPU") -> None:
        """Load the backend-specific model.  Must be implemented by subclasses."""
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
        """
        # Remove complete <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove unclosed <think> block (truncated generation)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _extract_function_calls(text: str) -> str:
        """Convert model output in 'commentary to=functions.X json{...}' format
        to Python-style function call(s) expected by the AST decoder.

        Handles one or more function calls in a single response.
        Falls through unchanged if the pattern is not found.
        """
        pattern = re.compile(
            r"to=functions\.([\w.]+)\s+json(\{.*?\})(?=\s*to=functions\.|\s*$)",
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
        return default_decode_ast_prompting(
            self._preprocess_result(result), language, has_tool_call_tag
        )

    @override
    def decode_execute(self, result, has_tool_call_tag):
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
            # Use up to 16384 tokens to accommodate reasoning models (e.g. Qwen3)
            # that generate a long <think> block before the actual answer.
            max_new_tokens = min(
                16384,
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
            max_new_tokens = min(16384, self.max_context_length - input_token_count - 2)

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
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        text: str = api_response["text"]
        # Strip thinking tags (e.g. Qwen3 reasoning models)
        text = self._strip_thinking_tags(text)

        # Try to extract a JSON list / dict of tool calls from the response
        tool_calls_json = None
        # Match first JSON array or object in the response
        json_match = re.search(r"(\[.*?\]|\{.*?\})", text, re.DOTALL)
        if json_match:
            try:
                tool_calls_json = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        if tool_calls_json is not None:
            if isinstance(tool_calls_json, dict):
                tool_calls_json = [tool_calls_json]
            # Normalise: each item should be {"name": ..., "arguments": {...}}
            model_responses = []
            tool_call_ids = []
            for idx, call in enumerate(tool_calls_json):
                name = call.get("name") or call.get("function", {}).get("name", "")
                arguments = call.get("arguments") or call.get("function", {}).get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                model_responses.append({name: arguments})
                tool_call_ids.append(f"call_{idx}")
        else:
            model_responses = text
            tool_call_ids = []

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": {
                "role": "assistant",
                "content": text,
            },
            "tool_call_ids": tool_call_ids,
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
