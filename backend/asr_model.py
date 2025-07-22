import asyncio

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel


class ASRModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.loading = False
        print("loading model...")
        self.model = EncDecHybridRNNTCTCBPEModel.restore_from(self.model_path)
        print("model has been loaded.")

    async def transcribe(self, audio_file):
        self.loading = True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.model.transcribe, audio_file)
            return result[0].text
        finally:
            self.loading = False
