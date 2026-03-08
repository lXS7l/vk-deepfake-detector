import os
import uuid
import shutil
from pathlib import Path
from typing import Literal, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from yt_dlp import YoutubeDL

# Импортируем универсальный анализатор
from analyzer.face_analyzer import DeepfakeAnalyzer

load_dotenv()

# Инициализируем анализатор (с аудио)
analyzer = DeepfakeAnalyzer(use_audio=True, audio_deep_model=False)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN")

if not all([SUPABASE_URL, SUPABASE_ANON_KEY, VK_ACCESS_TOKEN]):
    print("⚠️  ВНИМАНИЕ: Не все переменные окружения заданы.")

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AntiDeepfake VK Video Analyzer")


class VideoDownloadRequest(BaseModel):
    vk_url: str


class VideoInfo(BaseModel):
    video_id: str
    title: str
    duration: int
    local_path: str
    thumbnail: str | None = None


class AnalysisResult(BaseModel):
    video_id: str
    status: Literal["success", "warning", "error"]
    verdict: Literal["authentic", "suspicious", "deepfake", "insufficient_data"]
    message: str
    deepfake_probability: float
    details: dict


@app.get("/")
async def root():
    return {
        "message": "AntiDeepfake API работает",
        "env_check": {
            "supabase_url": bool(SUPABASE_URL),
            "vk_token": bool(VK_ACCESS_TOKEN),
        },
    }


@app.get("/test-vk")
async def test_vk_token():
    if not VK_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="VK_ACCESS_TOKEN не задан")

    url = "https://api.vk.com/method/users.get"
    params = {
        "access_token": VK_ACCESS_TOKEN,
        "v": "5.131",
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise HTTPException(status_code=400, detail=f"Ошибка VK API: {data['error']}")
        return {"status": "ok", "user": data["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось проверить токен: {str(e)}")


@app.post("/download-video", response_model=VideoInfo)
async def download_video(request: VideoDownloadRequest):
    """Скачивает видео из VK"""
    vk_url = request.vk_url
    if not vk_url.startswith(("https://vk.com/", "https://m.vk.com/", "https://vkvideo.ru")):
        raise HTTPException(status_code=400, detail="Некорректная ссылка VK")

    file_id = str(uuid.uuid4())
    output_template = str(TEMP_DIR / f"{file_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        # Для ускорения можно добавить внешний загрузчик, например aria2c
        # "external_downloader": "aria2c",
        # "external_downloader_args": {"aria2c": ["-x", "16", "-k", "1M"]},
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(vk_url, download=True)

            if info.get("requested_downloads"):
                local_path = info["requested_downloads"][0]["filepath"]
            else:
                ext = info.get("ext", "mp4")
                local_path = str(TEMP_DIR / f"{file_id}.{ext}")

            if not Path(local_path).exists():
                raise Exception("Файл не был создан")

            return VideoInfo(
                video_id=file_id,
                title=info.get("title", "Без названия"),
                duration=info.get("duration", 0),
                local_path=local_path,
                thumbnail=info.get("thumbnail"),
            )
    except Exception as e:
        for f in TEMP_DIR.glob(f"{file_id}.*"):
            f.unlink()
        raise HTTPException(status_code=500, detail=f"Ошибка скачивания видео: {str(e)}")


@app.post("/analyze/{video_id}", response_model=AnalysisResult)
async def analyze_video(video_id: str, background_tasks: BackgroundTasks):
    """
    Анализирует ранее скачанное видео по video_id.
    """
    video_files = list(TEMP_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail=f"Видео с ID {video_id} не найдено")

    video_path = str(video_files[0])

    try:
        # Запускаем полный анализ
        result = analyzer.analyze_video(video_path)
        result["video_id"] = video_id
        background_tasks.add_task(analyzer.cleanup_temp, video_path)
        return result
    except Exception as e:
        background_tasks.add_task(analyzer.cleanup_temp, video_path)
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


class AnalyzeRequest(BaseModel):
    vk_url: str


@app.post("/analyze-url", response_model=AnalysisResult)
async def analyze_by_url(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Скачивает видео по ссылке и сразу анализирует (лица + аудио).
    """
    vk_url = request.vk_url
    if not vk_url.startswith(("https://vk.com/", "https://m.vk.com/", "https://vkvideo.ru")):
        raise HTTPException(status_code=400, detail="Некорректная ссылка VK")

    file_id = str(uuid.uuid4())
    output_template = str(TEMP_DIR / f"{file_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        'external_downloader': 'aria2c',
        'external_downloader_args': {
            'aria2c': [
                '-x', '16',  # Количество соединений к серверу (потоков)
                '-k', '1M',  # Размер одного куска (1 МБ)
                '--min-split-size', '1M',
                '--max-connection-per-server', '16',  # Макс. соединений на сервер
                '--continue', 'true',
            ]
        },
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(vk_url, download=True)

            if info.get("requested_downloads"):
                local_path = info["requested_downloads"][0]["filepath"]
            else:
                ext = info.get("ext", "mp4")
                local_path = str(TEMP_DIR / f"{file_id}.{ext}")

            if not Path(local_path).exists():
                raise Exception("Файл не был создан")

            # Полный анализ (видео + аудио)
            result = analyzer.analyze_video(local_path)
            result["video_id"] = file_id

            # Удаляем видео после ответа
            background_tasks.add_task(analyzer.cleanup_temp, local_path)

            return result

    except Exception as e:
        for f in TEMP_DIR.glob(f"{file_id}.*"):
            f.unlink()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Отдельный эндпоинт для анализа только аудиофайла.
    """
    file_id = str(uuid.uuid4())
    file_path = TEMP_DIR / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Используем аудиоанализатор напрямую
        from analyzer.audio_analyzer import AudioDeepfakeAnalyzer
        audio_analyzer = AudioDeepfakeAnalyzer(use_deep_model=False)
        result = audio_analyzer.analyze_audio_file(str(file_path))
        result["file_id"] = file_id
        return result
    finally:
        if file_path.exists():
            file_path.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)