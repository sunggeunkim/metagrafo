from features.translate_speech.hermeneia_client import make_hermeneia_transcribe


class _Response:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def json(self) -> dict:
        return self._body


def test_posts_source_target_and_model() -> None:
    seen: dict = {}

    def post(url, *, files, data, headers):
        seen["url"] = url
        seen["pcm"] = files["audio"][1]
        seen["data"] = data
        seen["headers"] = headers
        return _Response(200, {"text": " Hello "})

    transcribe = make_hermeneia_transcribe(
        "https://hermeneia.stugen.net",
        "secret",
        "whisper",
        post=post,
    )
    text = transcribe(
        b"\x00\x01",
        language="ko",
        task="translate",
        initial_prompt="Jesus",
        condition_on_previous_text=True,
    )
    assert text == "Hello"
    assert seen["url"] == "https://hermeneia.stugen.net/v1/translations"
    assert seen["pcm"] == b"\x00\x01"
    assert seen["data"]["model"] == "whisper"
    assert seen["data"]["source_language"] == "ko"
    assert seen["data"]["target_language"] == "en"
    assert seen["data"]["initial_prompt"] == "Jesus"
    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_guest_mode_sends_english_to_english() -> None:
    seen: dict = {}

    def post(url, *, files, data, headers):
        seen["data"] = data
        seen["headers"] = headers
        return _Response(200, {"text": "Amen"})

    transcribe = make_hermeneia_transcribe("http://127.0.0.1:8080", "", "whisper", post=post)
    assert transcribe(b"\x00\x00", language="en", task="transcribe", initial_prompt="") == "Amen"
    assert seen["data"]["source_language"] == "en"
    assert seen["data"]["target_language"] == "en"
    assert "Authorization" not in seen["headers"]


def test_busy_and_unauthorized_return_empty_text() -> None:
    def post(url, *, files, data, headers):
        return _Response(503)

    transcribe = make_hermeneia_transcribe("http://h", "t", "seamless-m4t-v2", post=post)
    assert transcribe(b"\x00\x00", language="ko", task="translate", initial_prompt="") == ""

    def denied(url, *, files, data, headers):
        return _Response(401)

    transcribe = make_hermeneia_transcribe("http://h", "t", "whisper", post=denied)
    assert transcribe(b"\x00\x00", language="ko", task="translate", initial_prompt="") == ""
