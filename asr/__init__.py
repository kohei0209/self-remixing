# from .espnet_asr import ESPNetPretrainedASR
from .whisper_asr import WhisperASR


def call_asr_model(
    model: str,
    **kwargs,
):
    if "whisper" in model:
        return WhisperASR(model, **kwargs)
    else:
        raise NotImplementedError("Use whisper model")
        # return ESPNetPretrainedASR(model, **kwargs)
