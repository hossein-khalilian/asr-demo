import asyncio
import os

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel


class ASRModel:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.loading = False
        print("Loading model...")

        if os.path.exists(self.model_name_or_path):
            self.model = EncDecHybridRNNTCTCBPEModel.restore_from(
                restore_path=self.model_name_or_path
            )
        else:
            self.model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
                model_name=self.model_name_or_path
            )

        print("Model has been loaded.")

    async def transcribe(self, audio_file):
        self.loading = True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.model.transcribe, audio_file)
            return result[0].text
        finally:
            self.loading = False
