from typing import Optional

from bfcl_eval.model_handler.local_inference.base_openvino_handler import (
    BaseOpenVINOHandler,
)
from overrides import override


class OpenVINOOptimumHandler(BaseOpenVINOHandler):
    """
    Handler for OpenVINO IR models using the **optimum-intel** library
    (``OVModelForCausalLM``).

    Requirements
    ------------
    Install the optimum-intel package with the OpenVINO backend::

        pip install optimum[openvino]
        # or
        pip install optimum-intel openvino

    Model directory
    ---------------
    The model directory must contain:
      - OpenVINO IR model files: ``openvino_model.xml`` / ``openvino_model.bin``
        (produced by ``optimum-cli export openvino …`` or similar tools)
      - Tokenizer config files: ``tokenizer_config.json``, ``tokenizer.json``, etc.

    Pre-converted models are also available on Hugging Face under the
    ``OpenVINO/`` organisation (e.g. ``OpenVINO/Qwen2.5-7B-Instruct-int4-ov``).

    Usage example
    -------------
    ::

        bfcl generate \\
            --model OpenVINO/Qwen2.5-7B-Instruct-int4-ov-optimum \\
            --local-model-path /path/to/openvino_model_dir \\
            --openvino-device CPU \\
            --test-category simple

    Notes
    -----
    * ``export=False`` is passed to ``OVModelForCausalLM.from_pretrained`` so that
      the model is loaded from pre-exported IR files rather than being re-exported
      from PyTorch weights at runtime.
    * Temperature-based sampling is enabled only when ``temperature > 0.01``; below
      that threshold greedy decoding is used, which is consistent with the rest of
      the BFCL evaluation framework (default temperature = 0.001).
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
        self._ov_model = None

    @override
    def _load_model(
        self, model_path: str, device: str = "CPU", device_properties: Optional[dict] = None
    ) -> None:
        from optimum.intel import OVModelForCausalLM

        load_kwargs: dict = dict(
            export=False,  # load pre-converted OpenVINO IR files
            device=device,
            trust_remote_code=True,
        )
        if device_properties:
            print(f"[INFO] Applying OpenVINO device properties: {device_properties}")
            load_kwargs["ov_config"] = device_properties

        self._ov_model = OVModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    @override
    def _unload_model(self) -> None:
        self._ov_model = None

    @override
    def _generate(self, formatted_prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_length: int = inputs.input_ids.shape[1]

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature > 0.01:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["do_sample"] = True
        else:
            generate_kwargs["do_sample"] = False

        outputs = self._ov_model.generate(**generate_kwargs)

        # Decode only the newly generated tokens (exclude the prompt)
        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
