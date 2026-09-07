from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    translate_mode: str = "ko_to_en"
    audio_device_name: str = "ATEN_Stream_to_USB"
    audio_device_index: int | None = None
    vad_preroll_ms: int = 200
    vad_min_silence_ms: int = 800
    vad_max_utterance_s: float = 12
    vad_min_speech_ms: int = 250
    whisper_model: str | None = None
    whisper_device: str | None = None
    whisper_compute_type: str | None = None
    church_vocabulary_path: str = "church_vocabulary.txt"
    translate_queue_size: int = 4
