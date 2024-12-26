import io
import os
import sys
import zipfile
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Optional
import ChatTTS
from tools.audio import pcm_arr_to_mp3_view
from tools.logger import get_logger
import torch
from modelscope import snapshot_download
from pydantic import BaseModel
from tools.cfg import MODEL_DIR

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

logger = get_logger("Command")

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global chat

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    except Exception as e:
        logger.error(f"Failed to select device: {e}")
        device = torch.device("cpu")  # 默认回退到 CPU

    try:
        chattts_dir = snapshot_download('AI-ModelScope/ChatTTS', cache_dir=MODEL_DIR)
        chat = ChatTTS.Chat(get_logger("ChatTTS"))
        logger.info(f"当前设备类型:{device}")
        chat.load(source="custom", custom_path=chattts_dir, device=device, compile=True)
    except Exception as e:
        # 打印或记录错误信息
        logger.error("Error initializing ChatTTS.Chat:", e)
        raise


class ChatTTSParams(BaseModel):
    text: list[str]
    stream: bool = False
    lang: Optional[str] = None
    skip_refine_text: bool = False
    refine_text_only: bool = False
    use_decoder: bool = True
    do_text_normalization: bool = True
    do_homophone_replacement: bool = False
    params_refine_text: ChatTTS.Chat.RefineTextParams
    params_infer_code: ChatTTS.Chat.InferCodeParams


@app.post("/internal/generate/voice")
async def generate_voice(params: ChatTTSParams):
    logger.info("Text input: %s", str(params.text))

    # audio seed
    if params.params_infer_code.manual_seed is not None:
        torch.manual_seed(params.params_infer_code.manual_seed)
        params.params_infer_code.spk_emb = chat.sample_random_speaker()

    # text seed for text refining
    if params.params_refine_text:
        text = chat.infer(
            text=params.text, skip_refine_text=False, refine_text_only=True
        )
        logger.info(f"Refined text: {text}")
    else:
        # no text refining
        text = params.text

    logger.info("Use speaker:")
    logger.info(params.params_infer_code.spk_emb)

    logger.info("Start voice inference.")
    wavs = chat.infer(
        text=text,
        stream=params.stream,
        lang=params.lang,
        skip_refine_text=params.skip_refine_text,
        use_decoder=params.use_decoder,
        do_text_normalization=params.do_text_normalization,
        do_homophone_replacement=params.do_homophone_replacement,
        params_infer_code=params.params_infer_code,
        params_refine_text=params.params_refine_text,
    )
    logger.info("Inference completed.")

    # zip all of the audio files together
    buf = io.BytesIO()
    with zipfile.ZipFile(
            buf, "a", compression=zipfile.ZIP_DEFLATED, allowZip64=False
    ) as f:
        for idx, wav in enumerate(wavs):
            f.writestr(f"{idx}.mp3", pcm_arr_to_mp3_view(wav))
    logger.info("Audio generation successful.")
    buf.seek(0)

    response = StreamingResponse(buf, media_type="application/zip")
    response.headers["Content-Disposition"] = "attachment; filename=audio_files.zip"
    return response
