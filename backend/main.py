import os
import uuid
from pathlib import Path
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from yt_dlp import YoutubeDL
from analyzer.face_analyzer import DeepfakeAnalyzer
from fastapi import Query
from cache import compute_video_hash, get_cached_result, save_cached_result
import logging

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

analyzer = DeepfakeAnalyzer(use_audio=True)

VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN")

if not all([VK_ACCESS_TOKEN]):
    print("⚠️  ВНИМАНИЕ: Не все переменные окружения заданы.")

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

allowed_urls = ("https://vk.com/", "https://m.vk.com/", "https://vkvideo.ru")
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
    status: str
    verdict: str
    message: str
    deepfake_probability: float
    details: dict


@app.get("/")
async def root():
    return {
        "message": "AntiDeepfake API работает",
        "env_check": {
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
    vk_url = request.vk_url
    if not vk_url.startswith(allowed_urls):
        raise HTTPException(status_code=400, detail="Некорректная ссылка VK")
    file_id = str(uuid.uuid4())
    output_template = str(TEMP_DIR / f"{file_id}.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
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

@app.post("/analyze-url")
async def analyze_by_url(
    request: VideoDownloadRequest,
    background_tasks: BackgroundTasks,
    force_refresh: bool = Query(False, description="Принудительно пересчитать и обновить кэш")
):
    """
    Скачивает видео по ссылке, анализирует (лица + аудио) и кэширует результат.
    Если force_refresh=True, кэш игнорируется и перезаписывается.
    """
    vk_url = request.vk_url
    if not vk_url.startswith(allowed_urls):
        raise HTTPException(status_code=400, detail="Некорректная ссылка VK")

    file_id = str(uuid.uuid4())
    output_template = str(TEMP_DIR / f"{file_id}.%(ext)s")
    local_path = None

    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
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

            # Вычисляем хеш видео
            video_hash = compute_video_hash(local_path)

            # Проверяем кэш, если не принудительно
            if not force_refresh:
                cached = get_cached_result(video_hash)
                if cached:
                    logger.info(f"Возвращён кэшированный результат для {video_hash}")
                    # Планируем удаление временного файла
                    background_tasks.add_task(analyzer.cleanup_temp, local_path)
                    return cached

            # Если кэша нет или force_refresh=True — анализируем
            result = analyzer.analyze_video(local_path)
            result["video_id"] = file_id
            result["cached"] = False

            # Сохраняем в кэш
            save_cached_result(video_hash, result)

            return result

    except Exception as e:
        if local_path and Path(local_path).exists():
            background_tasks.add_task(analyzer.cleanup_temp, local_path)
        else:
            for f in TEMP_DIR.glob(f"{file_id}.*"):
                try:
                    f.unlink()
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

    finally:
        # Планируем удаление видео (если не произошло раньше)
        if local_path and Path(local_path).exists():
            background_tasks.add_task(analyzer.cleanup_temp, local_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)