from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    translate_mode: str = "ko_to_en"
    audio_device_name: str = "VB-Audio Virtual Cable"
    audio_device_index: int | None = None
    vad_preroll_ms: int = 200
    vad_min_silence_ms: int = 500
    vad_pauses: int = 1
    condition_on_previous_text: bool = True
    vad_max_utterance_s: float = 12
    vad_min_speech_ms: int = 250
    whisper_model: str | None = "large-v3"
    whisper_device: str | None = "cuda"
    whisper_compute_type: str | None = "float16"
    translate_queue_size: int = 4
    hermeneia_url: str = ""
    hermeneia_token: str = ""
    hermeneia_model: str = "whisper"
    hermeneia_timeout_s: float = 30.0
    caption_engine: str = "whisper"
    gemini_api_key: str = ""

    def gemini_live(self) -> bool:
        return self.caption_engine == "gemini_live" and bool(self.gemini_api_key.strip())
