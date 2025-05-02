import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torchaudio
from .vixtss_module import MODEL_DIR, SPEAKER_PATH, generate_voice


class TTSResponseModel(BaseModel):
    value: str


app = FastAPI()


lang_demo = "vi"
use_deepfilter = True
normalize_text = True


@app.get("/tts")
async def getTTS(tts_text: str):

    out_wav = generate_voice(
        lang_demo, tts_text, SPEAKER_PATH, use_deepfilter, normalize_text
    )
    buffer = io.BytesIO()
    torchaudio.save(buffer, out_wav, 24000, format="wav")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"},
    )
