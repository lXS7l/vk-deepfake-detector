from .audio_analyzer import AudioDeepfakeAnalyzer

def __init__(self, use_audio_analysis: bool = True):
    self.use_audio_analysis = use_audio_analysis
    if use_audio_analysis:
        self.audio_analyzer = AudioDeepfakeAnalyzer(use_deep_model=False)  # False для скорости