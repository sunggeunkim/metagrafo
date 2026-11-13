from pydantic_settings import BaseSettings, SettingsConfigDict

# How cleanup writes a Korean caption line until the booth saves its own style.
KOREAN_STYLE_DEFAULT = (
    "Write it as a sermon is spoken from the pulpit: formal 합니다체 (~합니다, ~입니까, "
    "~하십시오) for every ending, even in a fragment, never 해요체 or 반말: 했어요 → 했습니다, "
    "있어요 → 있습니다, 거예요 → 것입니다, 것이었죠 → 것이었습니다, and a trailing ~요 as in "
    "'곳에서요.' is dropped. Use honorifics (하나님께서 ~하셨습니다, 예수님께서) only for God and "
    "Jesus, where the line speaks of them; never make God or Jesus the subject of a sentence "
    "about anyone else. The speaker never honors their own title: 저는 청소년부 목사입니다, not "
    "목사님입니다. Where the speaker says \"you\" to the listeners, write 여러분 (여러분이, "
    "여러분은, 여러분의), never 당신 or 너; never add 여러분 or 성도 여러분 where the speaker "
    "addresses no one. A question to the listeners, such as \"okay?\" or \"right?\", stays a "
    "question, in 합니다체: 아시겠습니까?, 그렇지 않습니까? Use natural Korean word order, not "
    "the English speaker's: put time, place and duration phrases before the verb, and fold "
    "trailing fragments such as '지난달에.' or '3박 4일 동안.' into the sentence they belong to."
)

# Church vocabulary for any service, kept in either direction, until the booth saves its own
# glossary. One sermon's names and terms belong in its Additional context, not here.
GLOSSARY_DEFAULT = """God → 하나님 (never 신 or 하느님); the Lord → 주님; Jesus Christ → 예수 그리스도, 예수님; the Holy Spirit → 성령, 성령님
the Word, Scripture, the Bible → 말씀, 성경
grace → 은혜; salvation → 구원; faith → 믿음; repentance → 회개; sin → 죄; the cross → 십자가; resurrection → 부활; the gospel → 복음; eternal life → 영생
worship, the service → 예배; prayer → 기도; praise → 찬양; blessing → 복, 축복; Amen → 아멘; Hallelujah → 할렐루야
church members, brothers and sisters (only when the speaker addresses them) → 성도 여러분; the congregation → 성도들
pastor, elder, deacon, youth pastor (someone else's title) → 목사님, 장로님, 집사님, 청소년부 목사님
pastor, elder, deacon, youth pastor (the speaker's own title) → 목사, 장로, 집사, 청소년부 목사
sermon → 설교; Sunday → 주일; church → 교회
Bible books and references as in the 개역개정 Bible: Genesis → 창세기, Job → 욥기, Psalms → 시편, Romans → 로마서, John 3:16 → 요한복음 3장 16절"""

# Booth control's caption engines. The two on Hermeneia name the model the server runs.
CAPTION_ENGINES = ("gemini_live", "whisper_local", "whisper_hermeneia", "seamless_hermeneia")
# What booth control calls each engine.
ENGINE_NAMES = {
    "gemini_live": "Gemini live",
    "whisper_local": "Whisper on this PC",
    "whisper_hermeneia": "Whisper on Hermeneia",
    "seamless_hermeneia": "Seamless on Hermeneia",
}
HERMENEIA_MODELS = {"whisper_hermeneia": "whisper", "seamless_hermeneia": "seamless-m4t-v2"}


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
    # The translation often pauses longer than 1.2 s mid-sentence, which showed half sentences;
    # the open line shows on its own only once the speaker has stopped.
    whole_sentences_quiet_s: float = 5.0
    clean_up_wording: bool = True
    cleanup_model: str = "gemini-3.5-flash-lite"
    # 3.5 Flash Lite took 1–8 s a line (median ~3 s) on 2026-11-09; past this the line shows as heard.
    cleanup_timeout_s: float = 4.0
    # Background on this service for the cleanup model. Booth control sets it while Metagrafo
    # runs; every start is empty, replacing any CLEANUP_CONTEXT value from the environment.
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
    # The church's speech server. Whisper uses it once a token is saved; until then, this PC.
    # Booth control's "Whisper on this PC" sets USE_HERMENEIA=false, so the token stays saved.
    hermeneia_url: str = "https://hermeneia.stugen.net"
    hermeneia_token: str = ""
    hermeneia_model: str = "whisper"
    use_hermeneia: bool = True
    hermeneia_timeout_s: float = 30.0
    caption_engine: str = "whisper"
    gemini_api_key: str = ""
    youtube_caption_url: str = ""

    def gemini_live(self) -> bool:
        return self.caption_engine == "gemini_live" and bool(self.gemini_api_key.strip())

    def hermeneia(self) -> bool:
        """Whether Whisper runs on the Hermeneia server rather than this PC."""
        return self.use_hermeneia and bool(self.hermeneia_url.strip() and self.hermeneia_token.strip())

    def engine(self) -> str:
        """The booth control engine these settings run: one of CAPTION_ENGINES."""
        if self.gemini_live():
            return "gemini_live"
        if not self.hermeneia():
            return "whisper_local"
        if self.hermeneia_model == HERMENEIA_MODELS["seamless_hermeneia"]:
            return "seamless_hermeneia"
        return "whisper_hermeneia"
