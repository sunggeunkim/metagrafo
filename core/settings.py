from pydantic_settings import BaseSettings, SettingsConfigDict

# How cleanup writes a Korean caption line until the booth saves its own style.
KOREAN_STYLE_DEFAULT = (
    "Write it as a sermon is spoken from the pulpit: formal 합니다체 endings "
    "(~합니다, ~입니까, ~하십시오), never 해요체 or 반말, and honorifics for God and Jesus "
    "(하나님께서 ~하셨습니다, 예수님께서). Address the congregation as 여러분 (여러분이, 여러분은, 여러분의), never 당신 or 너."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    translate_mode: str = "ko_to_en"
    audio_device_name: str = "HDMI (Elgato 4K S)"
    audio_device_index: int | None = None
    vad_preroll_ms: int = 200
    vad_min_silence_ms: int = 500
    vad_pauses: int = 1
    condition_on_previous_text: bool = True
    whole_sentences: bool = True
    whole_sentences_quiet_s: float = 1.2
    clean_up_wording: bool = True
    cleanup_model: str = "gemini-3.5-flash-lite"
    # 3.5 Flash Lite took 1–8 s a line (median ~3 s) on 2026-11-09; past this the line shows as heard.
    cleanup_timeout_s: float = 4.0
    # Background on this service for the cleanup model. Booth control fills it from
    # cleanup-context.txt at startup, replacing any CLEANUP_CONTEXT value from the environment.
    cleanup_context: str = ""
    # Booth control fills it from korean-style.txt at startup (the default when there is none),
    # replacing any KOREAN_STYLE value from the environment. Empty means no Korean style rule.
    korean_style: str = KOREAN_STYLE_DEFAULT
    save_transcript: bool = False
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
    youtube_caption_url: str = ""

    def gemini_live(self) -> bool:
        return self.caption_engine == "gemini_live" and bool(self.gemini_api_key.strip())
