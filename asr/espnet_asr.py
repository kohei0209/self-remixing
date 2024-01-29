from espnet2.bin.asr_inference import Speech2Text


class ESPNetPretrainedASR:
    def __init__(
        self,
        model,
        **kwargs,
    ):
        self.model = Speech2Text.from_pretrained(model, **kwargs)

    def __call__(self, audio):
        nbests = self.model(audio)
        text, *_ = nbests[0]
        return text
