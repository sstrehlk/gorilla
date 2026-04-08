from bfcl_eval.model_handler.local_inference.base_openvino_handler import (
    BaseOpenVINOHandler,
)
from overrides import override


class OpenVINOGenAIHandler(BaseOpenVINOHandler):
    """
    Handler for OpenVINO IR models using the **openvino-genai** library
    (``openvino_genai.LLMPipeline``).

    Requirements
    ------------
    Install the openvino-genai package::

        pip install openvino-genai

    Model directory
    ---------------
    The model directory must contain:
      - OpenVINO IR model files compatible with openvino-genai
        (``openvino_model.xml`` / ``openvino_model.bin`` produced e.g. by
        ``optimum-cli export openvino …`` or Optimum Intel's ``OVModelForCausalLM``)
      - Tokenizer files compatible with the ``openvino_tokenizers`` extension, i.e.
        ``openvino_tokenizer.xml`` / ``openvino_detokenizer.xml`` **or** standard
        HuggingFace tokenizer files (``tokenizer_config.json``, etc.)

    The ``transformers`` tokenizer that lives alongside the IR files is used for
    token-count bookkeeping (prompt / output token counts) while the actual text
    generation is driven by ``openvino_genai.LLMPipeline``.

    Usage example
    -------------
    ::

        bfcl generate \\
            --model Qwen/Qwen2.5-7B-Instruct-OV-genai \\
            --local-model-path /path/to/openvino_model_dir \\
            --openvino-device CPU \\
            --test-category simple

    Notes
    -----
    * ``openvino_genai.LLMPipeline`` accepts a filesystem path; it cannot fetch
      models from the HuggingFace Hub, so ``--local-model-path`` must always be
      provided.
    * Temperature-based sampling is enabled only when ``temperature > 0.01``; below
      that threshold greedy decoding is used (consistent with the BFCL default of
      ``temperature = 0.001``).
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
        self._pipeline = None

    @override
    def _load_model(self, model_path: str, device: str = "CPU") -> None:
        import openvino_genai
        scheduler_config = openvino_genai.SchedulerConfig()
        scheduler_config.enable_prefix_caching = True
        self._pipeline = openvino_genai.LLMPipeline(
            model_path, device, config={"scheduler_config": scheduler_config}
        )

    @override
    def _unload_model(self) -> None:
        self._pipeline = None

    @override
    def _generate(self, formatted_prompt: str, max_new_tokens: int) -> str:
        import openvino_genai

        config = openvino_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens

        if self.temperature > 0.01:
            config.temperature = self.temperature
            config.do_sample = True
        else:
            config.do_sample = False

        # LLMPipeline.generate returns a str with only the newly generated text
        return self._pipeline.generate(formatted_prompt, config)
