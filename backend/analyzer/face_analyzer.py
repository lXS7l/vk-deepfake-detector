import os
import cv2
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
from keras_facenet import FaceNet
import logging
from typing import List, Dict, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeAnalyzer:
    def __init__(self):
        logger.info("Инициализация анализатора...")
        self.face_detector = MTCNN()
        self.facenet = FaceNet()
        self.frame_interval = 10
        self.face_similarity_threshold = 0.7
        logger.info("Анализатор готов")

    def extract_frames(self, video_path: str, max_frames: int = 100) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Всего кадров в видео: {total_frames}")

        frame_count = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Обработано кадров: {frame_count}/{total_frames}")

        cap.release()
        logger.info(f"Извлечено {len(frames)} кадров для анализа")
        return frames

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        try:
            return self.face_detector.detect_faces(frame)
        except Exception as e:
            logger.error(f"Ошибка детекции лиц: {e}")
            return []

    def get_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        try:
            embedding = self.facenet.embeddings([face_img])[0]
            return embedding
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга: {e}")
            return None

    def extract_face_image(self, frame: np.ndarray, face_box: List[int]) -> Optional[np.ndarray]:
        x, y, w, h = face_box
        padding = int(0.2 * w)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            return None
        face_img = cv2.resize(face_img, (160, 160))
        return face_img

    def analyze_video(self, video_path: str) -> Dict:
        logger.info(f"Начало анализа видео: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "status": "error",
                "verdict": "insufficient_data",
                "message": "Не удалось открыть видео",
                "deepfake_probability": 0.0,
                "details": {}
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Всего кадров: {total_frames}, FPS: {fps:.2f}")

        # Параметры поиска лиц
        search_interval = 15  # ищем лица с шагом 15 кадров (быстрее)
        min_faces_required = 5  # минимальное количество кадров с лицами для анализа
        max_frames_to_analyze = 100  # ограничим количество анализируемых кадров

        face_frames_indices = []  # номера кадров, где найдены лица
        embeddings = []  # сами эмбеддинги
        face_boxes = []  # координаты лиц для возможного использования

        frame_idx = 0
        while frame_idx < total_frames and len(embeddings) < max_frames_to_analyze:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция лиц на текущем кадре
            faces = self.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if faces:
                # Нашли лицо — запоминаем кадр
                face_frames_indices.append(frame_idx)
                # Берём самое крупное лицо
                main_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                face_box = main_face['box']
                face_img = self.extract_face_image(frame, face_box)
                if face_img is not None:
                    emb = self.get_face_embedding(face_img)
                    if emb is not None:
                        embeddings.append(emb)
                        face_boxes.append(face_box)

                # Переходим к следующему кадру с некоторым отступом, чтобы не дублировать одно лицо
                frame_idx += search_interval
            else:
                # Лиц нет — увеличиваем шаг (можно динамически менять)
                frame_idx += search_interval

            # Прогресс
            if frame_idx % (search_interval * 10) == 0:
                logger.info(f"Поиск лиц: обработано {frame_idx}/{total_frames} кадров, найдено {len(embeddings)} лиц")

        cap.release()

        # Проверяем, достаточно ли лиц
        if len(embeddings) < min_faces_required:
            return {
                "status": "warning",
                "verdict": "insufficient_data",
                "message": f"Найдено только {len(embeddings)} кадров с лицами (нужно минимум {min_faces_required})",
                "deepfake_probability": 0.0,
                "details": {
                    "faces_detected": len(embeddings),
                    "frames_scanned": frame_idx,
                    "total_frames": total_frames
                }
            }

        # Вычисляем схожесть между последовательными эмбеддингами
        similarities = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i] / np.linalg.norm(embeddings[i])
            b = embeddings[i + 1] / np.linalg.norm(embeddings[i + 1])
            sim = np.dot(a, b)
            similarities.append(sim)

        avg_similarity = float(np.mean(similarities))
        std_similarity = float(np.std(similarities))

        # Оценка вероятности дипфейка
        deepfake_score = 1.0 - avg_similarity
        if std_similarity > 0.15:
            deepfake_score += std_similarity * 0.5
        deepfake_score = min(1.0, max(0.0, deepfake_score))

        # Вердикт
        if deepfake_score < 0.3:
            verdict = "authentic"
            message = "Видео выглядит подлинным"
        elif deepfake_score < 0.6:
            verdict = "suspicious"
            message = "Обнаружены признаки обработки"
        else:
            verdict = "deepfake"
            message = "Высокая вероятность дипфейка"

        result = {
            "status": "success",
            "verdict": verdict,
            "message": message,
            "deepfake_probability": deepfake_score,
            "details": {
                "frames_analyzed": len(embeddings),
                "face_positions": face_boxes[:5],  # для примера первые 5
                "avg_face_similarity": avg_similarity,
                "similarity_std": std_similarity,
                "search_interval": search_interval
            }
        }

        logger.info(f"Анализ завершён. Вердикт: {verdict}, вероятность: {deepfake_score:.2f}")
        return result

    def cleanup_temp(self, video_path: str):
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Удалён временный файл: {video_path}")
        except Exception as e:
            logger.error(f"Ошибка удаления {video_path}: {e}")