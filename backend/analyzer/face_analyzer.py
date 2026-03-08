import os
import cv2
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
from keras_facenet import FaceNet
import logging
from typing import List, Dict, Optional

# Импортируем аудиоанализатор
from .audio_analyzer import AudioDeepfakeAnalyzer

logger = logging.getLogger(__name__)


class DeepfakeAnalyzer:
    """
    Универсальный анализатор видео на наличие дипфейков.
    Использует:
    - MTCNN для детекции лиц
    - FaceNet для получения эмбеддингов
    - AudioDeepfakeAnalyzer для анализа аудиодорожки
    """

    def __init__(self, use_audio: bool = True, audio_deep_model: bool = False):
        logger.info("Инициализация анализатора...")

        # Компоненты для видео
        self.face_detector = MTCNN()
        self.facenet = FaceNet()

        # Параметры видеоанализа
        self.frame_interval = 15  # шаг при поиске лиц
        self.min_faces_required = 5  # минимальное количество кадров с лицами
        self.max_frames_to_analyze = 100  # максимум кадров для анализа

        # Аудиоанализатор (если включён)
        self.use_audio = use_audio
        if use_audio:
            self.audio_analyzer = AudioDeepfakeAnalyzer(use_deep_model=audio_deep_model)

        logger.info("Анализатор готов")

    def extract_faces_from_video(self, video_path: str) -> Dict:
        """
        Извлекает лица из видео и возвращает эмбеддинги и метаданные.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {video_path}")
            return {
                "embeddings": [],
                "face_boxes": [],
                "frame_indices": [],
                "total_frames": 0
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Всего кадров в видео: {total_frames}")

        embeddings = []
        face_boxes = []
        frame_indices = []

        frame_idx = 0
        while frame_idx < total_frames and len(embeddings) < self.max_frames_to_analyze:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция лиц на кадре
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector.detect_faces(frame_rgb)

            if faces:
                # Берём самое крупное лицо
                main_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                x, y, w, h = main_face['box']

                # Вырезаем лицо с отступом
                padding = int(0.2 * w)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = w + 2 * padding
                h = h + 2 * padding

                face_img = frame_rgb[y:y + h, x:x + w]
                if face_img.size > 0:
                    face_img = cv2.resize(face_img, (160, 160))
                    emb = self.facenet.embeddings([face_img])[0]
                    embeddings.append(emb)
                    face_boxes.append([x, y, w, h])
                    frame_indices.append(frame_idx)

            frame_idx += self.frame_interval

            if frame_idx % (self.frame_interval * 20) == 0:
                logger.info(f"Поиск лиц: обработано {frame_idx}/{total_frames} кадров, найдено {len(embeddings)} лиц")

        cap.release()
        logger.info(f"Всего получено эмбеддингов лиц: {len(embeddings)}")

        return {
            "embeddings": embeddings,
            "face_boxes": face_boxes,
            "frame_indices": frame_indices,
            "total_frames": total_frames
        }

    def analyze_faces(self, embeddings: List[np.ndarray]) -> Dict:
        """
        Анализирует последовательность эмбеддингов лиц и возвращает
        оценку дипфейка на основе изменчивости.
        """
        if len(embeddings) < self.min_faces_required:
            return {
                "verdict": "insufficient_data",
                "probability": 0.0,
                "message": f"Найдено только {len(embeddings)} кадров с лицами",
                "details": {}
            }

        # Вычисляем попарную схожесть между соседними эмбеддингами
        similarities = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i] / np.linalg.norm(embeddings[i])
            b = embeddings[i + 1] / np.linalg.norm(embeddings[i + 1])
            sim = np.dot(a, b)
            similarities.append(sim)

        avg_similarity = float(np.mean(similarities))
        std_similarity = float(np.std(similarities))

        # Оценка вероятности дипфейка
        # В реальных видео лица меняются плавно -> высокая средняя схожесть
        # В дипфейках могут быть скачки -> низкая схожесть или высокое отклонение
        deepfake_score = 1.0 - avg_similarity
        if std_similarity > 0.15:
            deepfake_score += std_similarity * 0.5
        deepfake_score = min(1.0, max(0.0, deepfake_score))

        if deepfake_score < 0.3:
            verdict = "authentic"
            message = "Лица выглядят естественно"
        elif deepfake_score < 0.6:
            verdict = "suspicious"
            message = "Обнаружены аномалии в движениях лица"
        else:
            verdict = "deepfake"
            message = "Высокая вероятность подделки лица"

        return {
            "verdict": verdict,
            "probability": deepfake_score,
            "message": message,
            "details": {
                "embeddings_count": len(embeddings),
                "avg_similarity": avg_similarity,
                "similarity_std": std_similarity
            }
        }

    def analyze_video(self, video_path: str) -> Dict:
        """
        Полный анализ видео: лица + аудио.
        Возвращает объединённый результат.
        """
        # Шаг 1: анализ лиц
        face_data = self.extract_faces_from_video(video_path)
        face_result = self.analyze_faces(face_data["embeddings"])

        # Шаг 2: анализ аудио (если включён)
        audio_result = None
        if self.use_audio:
            audio_result = self.audio_analyzer.analyze_video_audio(video_path)

        # Шаг 3: комбинирование
        combined = self._combine_results(face_result, audio_result)

        # Добавляем технические детали
        combined["details"]["face_analysis"] = face_result["details"]
        if audio_result:
            combined["details"]["audio_analysis"] = {
                "verdict": audio_result.get("verdict"),
                "probability": audio_result.get("deepfake_probability"),
                "features": audio_result.get("details", {}).get("features", {})
            }

        combined["video_id"] = Path(video_path).stem

        return combined

    def _combine_results(self, face_result: Dict, audio_result: Optional[Dict]) -> Dict:
        """
        Комбинирует результаты анализа лица и аудио.
        Возвращает единый словарь с полями:
          status, verdict, message, deepfake_probability, details
        """
        # Если аудио отсутствует или не удалось
        if not audio_result or audio_result.get("status") != "success":
            # Используем только результат по лицу
            return {
                "status": "success" if face_result["verdict"] != "insufficient_data" else "warning",
                "verdict": face_result["verdict"] if face_result[
                                                         "verdict"] != "insufficient_data" else "insufficient_data",
                "message": face_result.get("message", "Недостаточно данных"),
                "deepfake_probability": face_result["probability"],
                "details": {}
            }

        # Если оба анализа успешны
        face_prob = face_result["probability"]
        audio_prob = audio_result["deepfake_probability"]

        # Веса: лица важнее (60%), аудио 40%
        combined_prob = 0.6 * face_prob + 0.4 * audio_prob

        # Определяем общий вердикт
        if combined_prob < 0.3:
            verdict = "authentic"
            message = "Видео и аудио выглядят подлинными"
        elif combined_prob < 0.6:
            verdict = "suspicious"
            message = "Обнаружены признаки обработки"
        else:
            verdict = "deepfake"
            message = "Высокая вероятность дипфейка"

        return {
            "status": "success",
            "verdict": verdict,
            "message": message,
            "deepfake_probability": combined_prob,
            "details": {}
        }

    def cleanup_temp(self, video_path: str):
        """Удаляет временный файл видео."""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Удалён временный файл: {video_path}")
        except Exception as e:
            logger.error(f"Ошибка удаления {video_path}: {e}")