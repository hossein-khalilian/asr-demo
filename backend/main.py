import asyncio
import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from asr_model import ASRModel

MODEL_PATH = "/home/user/.cache/nemo_experiments/Speech_To_Text_Finetuning/2025-07-22_06-58-20/checkpoints/Speech_To_Text_Finetuning.nemo"
MODEL_PATH = "/home/user/.cache/huggingface/hub/models--nvidia--stt_fa_fastconformer_hybrid_large/snapshots/249cf5bf70dda7220a60ddeeecff2f6aad8e1784/stt_fa_fastconformer_hybrid_large.nemo"
MODEL_NAME = "hsekhalilian/Speech_To_Text_Finetuning_01"
asr_model = ASRModel(MODEL_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        text = await asr_model.transcribe(tmp_path)
        return JSONResponse({"text": text})
    finally:
        os.remove(tmp_path)


@app.get("/status")
async def get_status():
    return JSONResponse({"loading": asr_model.loading})
