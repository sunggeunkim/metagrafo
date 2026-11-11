from pydantic_settings import BaseSettings, SettingsConfigDict

# How cleanup writes a Korean caption line until the booth saves its own style.
KOREAN_STYLE_DEFAULT = (
    "Write it as a sermon is spoken from the pulpit: formal 합니다체 endings "
    "(~합니다, ~입니까, ~하십시오), never 해요체 or 반말, and honorifics for God and Jesus "
    "(하나님께서 ~하셨습니다, 예수님께서). Address the congregation as 여러분 (여러분이, 여러분은, 여러분의), never 당신 or 너. "
    "Use natural Korean word order, not the English speaker's: put time, place and duration "
    "phrases before the verb, and fold trailing fragments such as '지난달에.' or "
    "'3박 4일 동안.' into the sentence they belong to."
)

# Church vocabulary for any service, kept in either direction, until the booth saves its own
# glossary. One sermon's names and terms belong in its Additional context, not here.
GLOSSARY_DEFAULT = """God → 하나님 (never 신 or 하느님); the Lord → 주님; Jesus Christ → 예수 그리스도, 예수님; the Holy Spirit → 성령, 성령님
the Word, Scripture, the Bible → 말씀, 성경
grace → 은혜; salvation → 구원; faith → 믿음; repentance → 회개; sin → 죄; the cross → 십자가; resurrection → 부활; the gospel → 복음; eternal life → 영생
worship, the service → 예배; prayer → 기도; praise → 찬양; blessing → 복, 축복; Amen → 아멘; Hallelujah → 할렐루야
church members, brothers and sisters → 성도 여러분; the congregation → 성도들
pastor → 목사님; elder → 장로님; deacon → 집사님; youth pastor → 청소년부 목사님
sermon → 설교; Sunday → 주일; church → 교회
Bible books and references as in the 개역개정 Bible: Genesis → 창세기, Job → 욥기, Psalms → 시편, Romans → 로마서, John 3:16 → 요한복음 3장 16절"""


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
    # Booth control fills it from glossary.txt at startup (the default when there is none),
    # replacing any GLOSSARY value from the environment. Empty means no glossary.
    glossary: str = GLOSSARY_DEFAULT
    save_transcript: bool = False
    # The overlay shows only the newest sentence, whole, until the next replaces it.
    one_sentence: bool = True
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
