"""vLLM engine construction for this machine.

On the RTX PRO 6000 (sm_120) vLLM 0.31 picks FlashInfer attention for gpt-oss, but
FlashInfer reports itself unavailable (no nvcc on PATH, no cubin package) and the
decode kernel fails at runtime. Triton attention supports gpt-oss's sinks and works on
every GPU we have tried, so it is the default. Set NATSTEER_VLLM_ATTENTION (e.g. to
FLASHINFER or FLASH_ATTN) on a machine where a faster backend works.
"""

import os


def make_llm(model: str, gpu_memory_utilization: float = 0.85, max_model_len: int = 8192, seed: int = 0, **kwargs):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM

    return LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, seed=seed, attention_backend=os.environ.get("NATSTEER_VLLM_ATTENTION", "TRITON_ATTN"), **kwargs)
