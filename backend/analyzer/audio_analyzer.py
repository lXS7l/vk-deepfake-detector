import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import webrtcvad

logger = logging.getLogger(__name__)


class AudioDeepfakeAnalyzer:
    """
    Анализатор аудио на наличие признаков синтеза речи.
    Использует комбинацию методов:
    1. Анализ спектральных аномалий через librosa
    2. Детекция пауз и артефактов через VAD
    3. MFCC-сравнение с эталоном (если доступно)
    4. Wav2Vec2 для глубокого анализа (опционально)
    """

    def __init__(self, use_deep_model: bool = False):
        logger.info("Инициализация аудиоанализатора...")

        # Параметры анализа
        self.sample_rate = 16000  # Гц, стандарт для речевых моделей
        self.frame_duration = 30  # мс для VAD
        self.vad = webrtcvad.Vad(2)  # Уровень агрессивности 2 (средний)

        # Глубокие модели (загружаются только при необходимости)
        self.use_deep_model = use_deep_model
        self.wav2vec_processor = None
        self.wav2vec_model = None

        if use_deep_model:
            self._load_deep_models()

        logger.info("Аудиоанализатор готов")

    def _load_deep_models(self):
        """Загружает Wav2Vec2 модель для глубокого анализа"""
        try:
            model_name = "facebook/wav2vec2-base-960h"
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.wav2vec_model.eval()
            logger.info("Wav2Vec2 модель загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки Wav2Vec2: {e}")
            self.use_deep_model = False

    def extract_audio_from_video(self, video_path: str, output_audio_path: Optional[str] = None) -> Optional[str]:
        """
        Извлекает аудиодорожку из видео с помощью ffmpeg.
        Возвращает путь к WAV-файлу.
        """
        if output_audio_path is None:
            video_name = Path(video_path).stem
            output_audio_path = str(Path(video_path).parent / f"{video_name}_audio.wav")

        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # без видео
                '-acodec', 'pcm_s16le',  # WAV формат
                '-ar', str(self.sample_rate),  # частота дискретизации
                '-ac', '1',  # моно
                '-y',  # перезаписать если существует
                output_audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Ошибка ffmpeg: {result.stderr}")
                return None

            if os.path.exists(output_audio_path):
                logger.info(f"Аудио извлечено: {output_audio_path}")
                return output_audio_path
            else:
                logger.error("Файл не создан после ffmpeg")
                return None

        except Exception as e:
            logger.error(f"Ошибка извлечения аудио: {e}")
            return None

    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Загружает аудиофайл и приводит к нужной частоте дискретизации.
        """
        try:
            # Загружаем с оригинальной частотой
            signal, sr = librosa.load(audio_path, sr=None)

            # Ресемплируем если нужно
            if sr != self.sample_rate:
                signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate

            logger.info(f"Аудио загружено: длительность {len(signal) / sr:.2f}с, частота {sr}Гц")
            return signal

        except Exception as e:
            logger.error(f"Ошибка загрузки аудио: {e}")
            return None

    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Извлекает спектральные признаки через librosa.
        Возвращает словарь с метриками, характерными для синтезированной речи.
        """
        features = {}

        try:
            # MFCC - основные коэффициенты
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = float(np.mean(mfcc))
            features['mfcc_std'] = float(np.std(mfcc))

            # Спектральный центроид (яркость звука)
            cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['centroid_mean'] = float(np.mean(cent))
            features['centroid_std'] = float(np.std(cent))

            # Спектральный контраст
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['contrast_mean'] = float(np.mean(contrast))

            # Zero-crossing rate (частота пересечения нуля)
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))

            # RMS energy (громкость)
            rms = librosa.feature.rms(y=audio)
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))

            # Спектральная полоса пропускания
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features['bandwidth_mean'] = float(np.mean(bandwidth))

            # Темповые характеристики (для обнаружения неестественного ритма)
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_regularity'] = float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0

            # Форманты (для обнаружения синтеза)
            try:
                # Простая оценка формант через LPC
                lpc_coeffs = librosa.lpc(audio, order=16)
                # Анализ формант можно углубить, но для прототипа достаточно
                features['lpc_energy'] = float(np.sum(np.abs(lpc_coeffs)))
            except:
                features['lpc_energy'] = 0.0

        except Exception as e:
            logger.error(f"Ошибка извлечения спектральных признаков: {e}")

        return features

    def analyze_pauses(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Анализирует паттерны пауз в речи.
        Синтезированная речь часто имеет неестественные паузы.
        """
        features = {}

        try:
            # Разбиваем на фреймы для VAD
            frame_length = int(sr * self.frame_duration / 1000)  # в сэмплах
            audio_bytes = (audio * 32768).astype(np.int16).tobytes()  # в байты для VAD

            speech_frames = 0
            total_frames = 0
            pause_lengths = []
            current_pause = 0

            for i in range(0, len(audio) - frame_length, frame_length):
                frame = audio_bytes[i * 2:(i + frame_length) * 2]  # 2 байта на сэмпл
                if len(frame) < frame_length * 2:
                    break

                is_speech = self.vad.is_speech(frame, sr)
                total_frames += 1

                if is_speech:
                    speech_frames += 1
                    if current_pause > 0:
                        pause_lengths.append(current_pause)
                        current_pause = 0
                else:
                    current_pause += 1

            # Добавляем последнюю паузу
            if current_pause > 0:
                pause_lengths.append(current_pause)

            # Статистика по паузам
            speech_percent = (speech_frames / total_frames) * 100 if total_frames > 0 else 0

            features['speech_percent'] = float(speech_percent)
            features['pause_count'] = len(pause_lengths)

            if pause_lengths:
                features['avg_pause_frames'] = float(np.mean(pause_lengths))
                features['pause_std'] = float(np.std(pause_lengths))
                features['max_pause'] = float(np.max(pause_lengths))
            else:
                features['avg_pause_frames'] = 0.0
                features['pause_std'] = 0.0
                features['max_pause'] = 0.0

            # Отношение речи к паузам
            if features['avg_pause_frames'] > 0:
                features['speech_pause_ratio'] = speech_percent / features['avg_pause_frames']
            else:
                features['speech_pause_ratio'] = speech_percent

        except Exception as e:
            logger.error(f"Ошибка анализа пауз: {e}")

        return features

    def analyze_with_wav2vec(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Использует Wav2Vec2 для глубокого анализа.
        Возвращает confidence и признаки.
        """
        if not self.use_deep_model:
            return {}

        features = {}

        try:
            # Подготовка входных данных
            inputs = self.wav2vec_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

            with torch.no_grad():
                logits = self.wav2vec_model(inputs.input_values).logits

            # Получаем предсказанные токены
            predicted_ids = torch.argmax(logits, dim=-1)

            # Анализируем распределение вероятностей
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1).values

            # Синтезированная речь часто имеет аномально высокие/низкие вероятности
            features['wav2vec_confidence_mean'] = float(torch.mean(max_probs).item())
            features['wav2vec_confidence_std'] = float(torch.std(max_probs).item())
            features['wav2vec_confidence_min'] = float(torch.min(max_probs).item())

            # Энтропия распределения (высокая энтропия может указывать на неопределенность)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            features['wav2vec_entropy_mean'] = float(torch.mean(entropy).item())

        except Exception as e:
            logger.error(f"Ошибка Wav2Vec2 анализа: {e}")

        return features

    def analyze_audio_file(self, audio_path: str) -> Dict:
        """
        Основной метод анализа аудиофайла.
        Возвращает словарь с результатами.
        """
        logger.info(f"Начало анализа аудио: {audio_path}")

        # 1. Загружаем аудио
        audio = self.load_audio(audio_path)
        if audio is None:
            return {
                "status": "error",
                "verdict": "insufficient_data",
                "message": "Не удалось загрузить аудиофайл",
                "deepfake_probability": 0.0,
                "details": {}
            }

        # 2. Извлекаем признаки
        spectral_features = self.extract_spectral_features(audio, self.sample_rate)
        pause_features = self.analyze_pauses(audio, self.sample_rate)
        deep_features = self.analyze_with_wav2vec(audio, self.sample_rate)

        # 3. Комбинируем все признаки
        all_features = {**spectral_features, **pause_features, **deep_features}

        # 4. Простая эвристика для оценки (в реальном проекте здесь была бы ML-модель)
        deepfake_score = self._calculate_deepfake_score(all_features)

        # 5. Формируем вердикт
        if deepfake_score < 0.3:
            verdict = "authentic"
            message = "Аудио выглядит подлинным"
        elif deepfake_score < 0.6:
            verdict = "suspicious"
            message = "Обнаружены признаки синтеза речи"
        else:
            verdict = "deepfake"
            message = "Высокая вероятность синтезированной речи"

        result = {
            "status": "success",
            "verdict": verdict,
            "message": message,
            "deepfake_probability": deepfake_score,
            "details": {
                "features": all_features,
                "feature_count": len(all_features)
            }
        }

        logger.info(f"Анализ аудио завершён. Вердикт: {verdict}, вероятность: {deepfake_score:.2f}")
        return result

    def _calculate_deepfake_score(self, features: Dict) -> float:
        """
        Вычисляет вероятность дипфейка на основе эвристик.
        В реальном проекте здесь должна быть обученная модель.
        """
        score = 0.5  # начальное значение

        if not features:
            return 0.0

        # Признаки синтезированной речи:
        # 1. Слишком низкая вариативность MFCC
        if 'mfcc_std' in features:
            if features['mfcc_std'] < 5.0:
                score += 0.15
            elif features['mfcc_std'] > 15.0:
                score -= 0.1

        # 2. Аномальный ZCR
        if 'zcr_mean' in features:
            if features['zcr_mean'] > 0.1:  # слишком много высоких частот
                score += 0.1

        # 3. Неестественные паузы
        if 'pause_std' in features:
            if features['pause_std'] < 0.5:  # слишком регулярные паузы
                score += 0.2
            elif features['pause_std'] > 3.0:  # слишком хаотичные
                score += 0.1

        # 4. Если есть данные от Wav2Vec2
        if 'wav2vec_confidence_std' in features:
            if features['wav2vec_confidence_std'] < 0.1:  # слишком уверенно
                score += 0.15

        # Нормализуем
        score = min(1.0, max(0.0, score))
        return score

    def analyze_video_audio(self, video_path: str) -> Dict:
        """
        Извлекает аудио из видео и анализирует его.
        Удаляет временный аудиофайл после анализа.
        """
        # Извлекаем аудио
        audio_path = self.extract_audio_from_video(video_path)
        if audio_path is None:
            return {
                "status": "error",
                "verdict": "insufficient_data",
                "message": "Не удалось извлечь аудио из видео",
                "deepfake_probability": 0.0,
                "details": {}
            }

        try:
            # Анализируем
            result = self.analyze_audio_file(audio_path)
            return result
        finally:
            # Чистим временный файл
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info(f"Удалён временный аудиофайл: {audio_path}")
            except Exception as e:
                logger.error(f"Ошибка удаления {audio_path}: {e}")

    def cleanup_temp(self, audio_path: str):
        """Удаляет временный файл"""
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Удалён временный файл: {audio_path}")
        except Exception as e:
            logger.error(f"Ошибка удаления {audio_path}: {e}")