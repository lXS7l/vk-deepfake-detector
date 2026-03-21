import os
import cv2
import numpy as np
from pathlib import Path
from keras_facenet import FaceNet
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DeepfakeAnalyzer:
    def __init__(self, use_audio: bool = False, audio_deep_model: bool = False):
        logger.info("Инициализация анализатора...")
        self.face_detector = self._init_opencv_face_detector()
        self.facenet = FaceNet()
        self.frame_interval = 15
        self.min_faces_required = 5
        self.max_frames_to_analyze = 100
        self.use_audio = use_audio
        if use_audio:
            from .audio_analyzer import AudioDeepfakeAnalyzer
            self.audio_analyzer = AudioDeepfakeAnalyzer(use_deep_model=audio_deep_model)
        logger.info("Анализатор готов")

    def _init_opencv_face_detector(self):
        """Загружает детектор лиц OpenCV DNN из локальных файлов."""
        model_dir = Path(__file__).parent / "models"
        prototxt_path = model_dir / "deploy.prototxt"
        model_path = model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

        if not prototxt_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Файлы модели не найдены в {model_dir}. "
                "Пожалуйста, скачайте их вручную: "
                "1. deploy.prototxt (https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)\n"
                "2. res10_300x300_ssd_iter_140000_fp16.caffemodel (https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel)"
            )

        net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
        return net

    def extract_faces_from_video(self, video_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {video_path}")
            return {"embeddings": [], "face_boxes": [], "frame_indices": [], "total_frames": 0}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Всего кадров в видео: {total_frames}")

        embeddings = []
        face_boxes = []
        frame_indices = []
        frame_idx = 0

        try:
            while frame_idx < total_frames and len(embeddings) < self.max_frames_to_analyze:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                if frame is None or frame.size == 0:
                    frame_idx += self.frame_interval
                    continue

                h, w = frame.shape[:2]
                if h < 48 or w < 48:
                    frame_idx += self.frame_interval
                    continue

                # Детекция лиц через OpenCV DNN
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()

                best_confidence = 0.0
                best_box = None
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5 and confidence > best_confidence:
                        best_confidence = confidence
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, x2, y2) = box.astype(int)
                        best_box = (x, y, x2 - x, y2 - y)

                if best_box is not None:
                    x, y, w_face, h_face = best_box
                    # Корректируем границы
                    x = max(0, x)
                    y = max(0, y)
                    w_face = min(w_face, w - x)
                    h_face = min(h_face, h - y)
                    if w_face <= 0 or h_face <= 0:
                        frame_idx += self.frame_interval
                        continue

                    padding = int(0.2 * w_face)
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w_face = w_face + 2 * padding
                    h_face = h_face + 2 * padding

                    face_img = frame[y:y+h_face, x:x+w_face]
                    if face_img.size == 0:
                        frame_idx += self.frame_interval
                        continue

                    face_img = cv2.resize(face_img, (160, 160))
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    emb = self.facenet.embeddings([face_rgb])[0]
                    embeddings.append(emb)
                    face_boxes.append([x, y, w_face, h_face])
                    frame_indices.append(frame_idx)

                frame_idx += self.frame_interval

        finally:
            cap.release()

        logger.info(f"Всего получено эмбеддингов лиц: {len(embeddings)}")
        return {
            "embeddings": embeddings,
            "face_boxes": face_boxes,
            "frame_indices": frame_indices,
            "total_frames": total_frames
        }

    def analyze_faces(self, embeddings: List[np.ndarray]) -> Dict:
        if len(embeddings) < self.min_faces_required:
            return {
                "verdict": "insufficient_data",
                "probability": 0.0,
                "message": f"Найдено только {len(embeddings)} кадров с лицами",
                "details": {}
            }

        similarities = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i] / np.linalg.norm(embeddings[i])
            b = embeddings[i+1] / np.linalg.norm(embeddings[i+1])
            sim = np.dot(a, b)
            similarities.append(sim)

        avg_similarity = float(np.mean(similarities))
        std_similarity = float(np.std(similarities))

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
        face_data = self.extract_faces_from_video(video_path)
        face_result = self.analyze_faces(face_data["embeddings"])

        audio_result = None
        if self.use_audio:
            audio_result = self.audio_analyzer.analyze_video_audio(video_path)

        combined = self._combine_results(face_result, audio_result)
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
        if not audio_result or audio_result.get("status") != "success":
            return {
                "status": "success" if face_result["verdict"] != "insufficient_data" else "warning",
                "verdict": face_result["verdict"] if face_result["verdict"] != "insufficient_data" else "insufficient_data",
                "message": face_result.get("message", "Недостаточно данных"),
                "deepfake_probability": face_result["probability"],
                "details": {}
            }

        face_prob = face_result["probability"]
        audio_prob = audio_result["deepfake_probability"]
        combined_prob = 0.6 * face_prob + 0.4 * audio_prob

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
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Удалён временный файл: {video_path}")
        except Exception as e:
            logger.error(f"Ошибка удаления {video_path}: {e}")