from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


def make_hermeneia_transcribe(
    url: str,
    token: str,
    model: str,
    *,
    timeout_s: float = 30.0,
    post=None,
):
    """PCM in, text out. Source and target come from the caller's language and task."""

    endpoint = url.rstrip("/") + "/v1/translations"

    def transcribe(
        pcm_s16le: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        target = "en" if task == "translate" else language
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        files = {"audio": ("chunk.pcm", pcm_s16le, "application/octet-stream")}
        data = {
            "model": model,
            "source_language": language,
            "target_language": target,
            "sample_rate": "16000",
            "encoding": "s16le",
            "initial_prompt": initial_prompt or "",
            "condition_on_previous_text": "true" if condition_on_previous_text else "false",
        }
        try:
            if post is not None:
                response = post(endpoint, files=files, data=data, headers=headers)
            else:
                response = httpx.post(
                    endpoint,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=timeout_s,
                )
        except httpx.TimeoutException:
            logger.warning("hermeneia timed out")
            return ""
        except httpx.HTTPError:
            logger.exception("hermeneia request failed")
            return ""
        if response.status_code != 200:
            logger.warning("hermeneia returned %s", response.status_code)
            return ""
        body = response.json()
        return (body.get("text") or "").strip()

    return transcribe
