import sqlite3
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "cache.db"

def get_db_connection():
    """Возвращает соединение с SQLite базой данных."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Создаёт таблицу, если её нет."""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analysis_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_hash TEXT UNIQUE NOT NULL,
            result_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def compute_video_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Вычисляет SHA256 хеш видеофайла (первые 5 МБ для скорости).
    """
    sha = hashlib.sha256()
    size = os.path.getsize(file_path)
    # Берём первые 5 МБ (если файл меньше — весь)
    read_limit = min(size, 5 * 1024 * 1024)
    with open(file_path, 'rb') as f:
        bytes_read = 0
        while bytes_read < read_limit:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
            bytes_read += len(chunk)
    return sha.hexdigest()

def get_cached_result(video_hash: str) -> dict | None:
    """Возвращает кэшированный результат или None."""
    conn = get_db_connection()
    cur = conn.execute(
        "SELECT result_json FROM analysis_cache WHERE video_hash = ?",
        (video_hash,)
    )
    row = cur.fetchone()
    conn.close()
    if row:
        # Обновляем last_accessed
        conn = get_db_connection()
        conn.execute(
            "UPDATE analysis_cache SET last_accessed = CURRENT_TIMESTAMP WHERE video_hash = ?",
            (video_hash,)
        )
        conn.commit()
        conn.close()
        return json.loads(row["result_json"])
    return None

def save_cached_result(video_hash: str, result: dict):
    """Сохраняет результат в кэш."""
    conn = get_db_connection()
    conn.execute(
        "INSERT OR REPLACE INTO analysis_cache (video_hash, result_json) VALUES (?, ?)",
        (video_hash, json.dumps(result, ensure_ascii=False))
    )
    conn.commit()
    conn.close()
    logger.info(f"Результат сохранён в кэш для {video_hash}")

def delete_cached_result(video_hash: str):
    """Удаляет запись из кэша."""
    conn = get_db_connection()
    conn.execute("DELETE FROM analysis_cache WHERE video_hash = ?", (video_hash,))
    conn.commit()
    conn.close()
    logger.info(f"Кэш удалён для {video_hash}")

# Инициализируем базу при первом импорте
init_db()