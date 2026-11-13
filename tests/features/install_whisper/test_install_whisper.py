import time

import pytest
from fastapi.testclient import TestClient

from core.settings import Settings
from features.install_whisper import check, installer
from features.install_whisper.check import Gpu
from main import create_app

RTX_3060_LAPTOP = Gpu(name="NVIDIA GeForce RTX 3060 Laptop GPU", total_gb=6.0, free_gb=5.4, driver="596.08")


class _ThisPC:
    """A booth with a stand-in GPU, disk, pip and model download."""

    def __init__(
        self,
        tmp_path,
        monkeypatch,
        *,
        gpu: Gpu | None = RTX_3060_LAPTOP,
        disk_gb: float = 120.0,
        pip_ok: bool = True,
        pip_s: float = 0.0,
        installed: bool = False,
    ) -> None:
        for name in ("CAPTION_ENGINE", "HERMENEIA_TOKEN", "HERMENEIA_MODEL", "USE_HERMENEIA", "WHISPER_MODEL"):
            monkeypatch.delenv(name, raising=False)
        self.installed = installed
        self.pip_runs: list[str] = []
        self.downloads: list[str] = []
        monkeypatch.setattr(check, "read_gpu", lambda: gpu)
        monkeypatch.setattr(check, "disk_free_gb", lambda _path: disk_gb)
        monkeypatch.setattr("features.translate_speech.whisper_asr.installed", lambda: self.installed)
        monkeypatch.setattr(
            "features.translate_speech.whisper_asr.make_transcribe",
            lambda _profile: lambda _pcm, **_kw: "Hello from this PC.",
        )
        monkeypatch.setattr(
            "features.translate_speech.hermeneia_client.make_hermeneia_transcribe",
            lambda *_a, **_k: lambda _pcm, **_kw: "Hello from Whisper on Hermeneia.",
        )

        def run_pip(log) -> bool:
            self.pip_runs.append(installer.REQUIREMENTS.name)
            time.sleep(pip_s)
            log("Successfully installed faster-whisper" if pip_ok else "ERROR: No matching distribution")
            if pip_ok:
                self.installed = True
            return pip_ok

        monkeypatch.setattr(installer, "run_pip", run_pip)
        monkeypatch.setattr(installer, "download_model", lambda model: self.downloads.append(model))
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        self.app = create_app(
            settings=Settings(caption_engine="whisper", hermeneia_token="tok-42", _env_file=env_file),
            env_file=env_file,
            load_whisper=True,
            list_input_devices=lambda: [],
        )

    def wait_for_install(self, client: TestClient) -> dict:
        for _ in range(300):
            status = client.get("/whisper-install").json()
            if status["state"] != "installing":
                return status
            time.sleep(0.01)
        raise AssertionError("install never finished")


def test_the_check_names_the_gpu_the_model_and_what_the_install_downloads(tmp_path, monkeypatch) -> None:
    pc = _ThisPC(tmp_path, monkeypatch)
    with TestClient(pc.app) as client:
        result = client.get("/whisper-check").json()
    assert result["verdict"] == "tight"
    assert result["can_install"] is True
    assert result["gpu"] == {
        "name": "NVIDIA GeForce RTX 3060 Laptop GPU", "total_gb": 6.0, "free_gb": 5.4, "driver": "596.08"
    }
    assert result["model"] == "large-v3"
    message = result["message"]
    assert "NVIDIA GeForce RTX 3060 Laptop GPU · 6.0 GB, 5.4 GB free now · driver 596.08" in message
    assert "large-v3" in message
    assert "about 4.5 GB" in message
    assert "120 GB free" in message


def test_a_gpu_with_room_to_spare_fits(tmp_path, monkeypatch) -> None:
    pc = _ThisPC(tmp_path, monkeypatch, gpu=Gpu(name="NVIDIA GeForce RTX 4070", total_gb=12.0, free_gb=10.5, driver="560.94"))
    with TestClient(pc.app) as client:
        result = client.get("/whisper-check").json()
    assert (result["verdict"], result["can_install"]) == ("fits", True)


def test_a_gpu_with_too_little_free_beside_obs_is_tight_but_can_install(tmp_path, monkeypatch) -> None:
    pc = _ThisPC(tmp_path, monkeypatch, gpu=Gpu(name="NVIDIA GeForce RTX 3060", total_gb=8.0, free_gb=3.0, driver="560.94"))
    with TestClient(pc.app) as client:
        result = client.get("/whisper-check").json()
    assert (result["verdict"], result["can_install"]) == ("tight", True)
    assert "OBS" in result["message"]


@pytest.mark.parametrize(
    ("gpu", "disk_gb", "verdict", "says"),
    [
        (None, 120.0, "no_gpu", "No NVIDIA GPU"),
        (Gpu(name="NVIDIA GeForce GTX 1060", total_gb=6.0, free_gb=5.5, driver="472.12"), 120.0, "old_driver", "527.41"),
        (Gpu(name="NVIDIA GeForce GTX 1650", total_gb=4.0, free_gb=3.6, driver="560.94"), 120.0, "too_small", "4.5 GB"),
        (RTX_3060_LAPTOP, 3.0, "no_disk", "Free up"),
    ],
)
def test_the_check_refuses_an_install_that_cannot_run_here(tmp_path, monkeypatch, gpu, disk_gb, verdict, says) -> None:
    pc = _ThisPC(tmp_path, monkeypatch, gpu=gpu, disk_gb=disk_gb)
    with TestClient(pc.app) as client:
        result = client.get("/whisper-check").json()
        assert (result["verdict"], result["can_install"]) == (verdict, False)
        assert says in result["message"]
        response = client.post("/whisper-install")
        assert response.status_code == 409
        assert response.json()["detail"] == result["message"]
    assert pc.pip_runs == []


def test_the_check_says_when_whisper_is_already_installed(tmp_path, monkeypatch) -> None:
    pc = _ThisPC(tmp_path, monkeypatch, installed=True)
    with TestClient(pc.app) as client:
        result = client.get("/whisper-check").json()
        assert (result["verdict"], result["can_install"]) == ("installed", False)
        assert "Whisper on this PC" in result["message"]
        assert client.post("/whisper-install").status_code == 409


def test_installing_whisper_turns_on_whisper_on_this_pc(tmp_path, monkeypatch) -> None:
    pc = _ThisPC(tmp_path, monkeypatch)
    with TestClient(pc.app) as client:
        assert client.get("/mode").json()["whisper_local_ready"] is False
        assert client.get("/whisper-install").json()["state"] == "idle"
        response = client.post("/whisper-install")
        assert response.status_code == 202
        status = pc.wait_for_install(client)
        assert status["state"] == "done"
        assert pc.pip_runs == ["requirements-whisper.txt"]
        # The model is downloaded, not loaded: the GPU stays with OBS until Whisper is picked.
        assert pc.downloads == ["large-v3"]
        assert client.get("/mode").json()["whisper_local_ready"] is True
        assert client.put("/caption-engine", json={"engine": "whisper_local"}).status_code == 200


def test_a_failed_install_says_so_and_leaves_whisper_off(tmp_path, monkeypatch) -> None:
    pc = _ThisPC(tmp_path, monkeypatch, pip_ok=False)
    with TestClient(pc.app) as client:
        assert client.post("/whisper-install").status_code == 202
        status = pc.wait_for_install(client)
        assert status["state"] == "failed"
        assert "Log" in status["error"]
        assert pc.downloads == []
        assert client.get("/mode").json()["whisper_local_ready"] is False


def test_a_second_install_click_while_installing_is_refused(tmp_path, monkeypatch) -> None:
    pc = _ThisPC(tmp_path, monkeypatch, pip_s=0.3)
    with TestClient(pc.app) as client:
        assert client.post("/whisper-install").status_code == 202
        response = client.post("/whisper-install")
        assert response.status_code == 409
        assert "already installing" in response.json()["detail"]
        assert pc.wait_for_install(client)["state"] == "done"
    assert pc.pip_runs == ["requirements-whisper.txt"]


def test_pip_runs_on_the_console_python_beside_pythonw(tmp_path) -> None:
    (tmp_path / "pythonw.exe").write_bytes(b"")
    (tmp_path / "python.exe").write_bytes(b"")
    assert installer.pip_python(str(tmp_path / "pythonw.exe")) == str(tmp_path / "python.exe")
    assert installer.pip_python(str(tmp_path / "python.exe")) == str(tmp_path / "python.exe")


def test_installed_packages_are_pinned_so_pip_leaves_open_files_alone() -> None:
    freeze = "fastapi==0.136.1\nnumpy==2.3.4\npip==25.2\n-e file:///c/metagrafo\nmylib @ file:///c/wheels/mylib.whl\n"
    assert installer.constraints(freeze) == "fastapi==0.136.1\nnumpy==2.3.4\n"


def test_the_driver_version_is_compared_as_numbers() -> None:
    assert check.driver_ok("596.08", "win32")
    assert check.driver_ok("527.41", "win32")
    assert not check.driver_ok("527.4", "win32")
    assert not check.driver_ok("472.12", "win32")
    assert check.driver_ok("550.54.15", "linux")
    assert not check.driver_ok("garbage", "win32")
