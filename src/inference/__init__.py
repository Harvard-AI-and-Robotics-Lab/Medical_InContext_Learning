from .output_parser import OutputParser, ClassificationParsedOutput, VQAParsedOutput


def MLLMClient(*args, **kwargs):
    from .mllm_client import MLLMClient as _MLLMClient
    return _MLLMClient(*args, **kwargs)


def HFLocalClient(*args, **kwargs):
    from .hf_local_client import HFLocalClient as _HFLocalClient
    return _HFLocalClient(*args, **kwargs)
