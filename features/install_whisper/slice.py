from __future__ import annotations

import asyncio
import importlib
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from core.settings import Settings
from features.install_whisper import check, installer

logger = logging.getLogger(__name__)


class InstallWhisperSlice:
    """Checks this PC for Whisper and installs it in the background, one install at a time."""

    def __init__(self, app: FastAPI, settings: Settings) -> None:
        self._app = app
        self._settings = settings
        self.state = "idle"
        self.step = ""
        self.error: str | None = None
        self._task: asyncio.Task[None] | None = None

    def profile(self) -> tuple[str, str]:
        """The model and compute type Whisper on this PC would load."""
        profile = getattr(self._app.state, "whisper_profile", None)
        if profile is not None:
            return profile.model, profile.compute_type
        return self._settings.whisper_model or "large-v3", self._settings.whisper_compute_type or "float16"

    def status(self) -> dict[str, object]:
        return {"state": self.state, "step": self.step, "error": self.error}

    async def check(self) -> dict[str, object]:
        # nvidia-smi takes up to a second; keep it off the event loop.
        return await asyncio.to_thread(check.check_this_pc, *self.profile())

    async def install(self) -> None:
        if self.state == "installing":
            raise HTTPException(status_code=409, detail="Whisper is already installing.")
        result = await self.check()
        if not result["can_install"]:
            raise HTTPException(status_code=409, detail=result["message"])
        self.state, self.error = "installing", None
        self._task = asyncio.create_task(self._install(), name="install-whisper")

    async def _install(self) -> None:
        model, _compute = self.profile()
        try:
            self.step = "Installing Whisper's packages. The progress shows under Log."
            logger.info("Installing Whisper: pip install -r %s", installer.REQUIREMENTS.name)
            if not await asyncio.to_thread(installer.run_pip, lambda line: logger.info("pip: %s", line)):
                self._fail("pip could not install Whisper. The reason is at the end of the Log.")
                return
            importlib.invalidate_caches()
            self.step = f"Downloading the {model} model."
            logger.info("Downloading the Whisper %s model", model)
            await asyncio.to_thread(installer.download_model, model)
        except asyncio.CancelledError:
            installer.stop()
            self._fail("The install stopped with Metagrafo. Check and install again.")
            raise
        except Exception as error:
            logger.exception("Whisper install failed")
            self._fail(f"The install failed ({error}). The reason is in the Log.")
            return
        self.state, self.step = "done", ""
        logger.info("Whisper is installed. Pick Whisper on this PC on booth control.")

    def _fail(self, error: str) -> None:
        self.state, self.step, self.error = "failed", "", error

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            installer.stop()
            self._task.cancel()


def register(app: FastAPI, settings: Settings) -> InstallWhisperSlice:
    slice_ = InstallWhisperSlice(app, settings)

    @app.get("/whisper-check")
    async def whisper_check() -> dict[str, object]:
        return await slice_.check()

    @app.get("/whisper-install")
    async def whisper_install_status() -> dict[str, object]:
        return slice_.status()

    @app.post("/whisper-install")
    async def whisper_install() -> JSONResponse:
        await slice_.install()
        return JSONResponse(slice_.status(), status_code=202)

    return slice_
