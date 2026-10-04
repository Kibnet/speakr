"""ASR_MIN_SPEAKERS / ASR_MAX_SPEAKERS with ASR_BASE_URL alone (#415).

The ASR connector is chosen from ASR_BASE_URL, and the docs mark the old
USE_ASR_ENDPOINT flag as deprecated, but the speaker-count defaults were only
read when that flag was set. An installation set up as documented sent no
speaker counts to WhisperX, also for the watch folder.

app_config reads the environment at import, so each case imports it in a
fresh interpreter.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = (
    "import json, src.config.app_config as c; "
    "print(json.dumps([c.ASR_MIN_SPEAKERS, c.ASR_MAX_SPEAKERS]))"
)


def _read(**env):
    clean = {k: v for k, v in os.environ.items()
             if k not in ("USE_ASR_ENDPOINT", "ASR_BASE_URL", "TRANSCRIPTION_CONNECTOR",
                          "ASR_MIN_SPEAKERS", "ASR_MAX_SPEAKERS")}
    clean.update(env)
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, env=clean,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("env", [
    {"ASR_BASE_URL": "http://whisperx:9000"},                              # as documented
    {"TRANSCRIPTION_CONNECTOR": "asr_endpoint", "ASR_BASE_URL": "http://whisperx:9000"},
    {"USE_ASR_ENDPOINT": "true", "ASR_BASE_URL": "http://whisperx:9000"},  # legacy flag
])
def test_speaker_counts_are_read_whenever_the_asr_connector_is_active(env):
    assert _read(ASR_MIN_SPEAKERS="2", ASR_MAX_SPEAKERS="4", **env) == ["2", "4"]


def test_speaker_counts_are_ignored_without_an_asr_connector():
    assert _read(ASR_MIN_SPEAKERS="2", ASR_MAX_SPEAKERS="4") == [None, None]


def test_the_resolver_hands_the_env_defaults_to_every_path(monkeypatch):
    import src.config.app_config as c
    from src.app import app
    from src.services.transcription_defaults import resolve_transcription_params
    monkeypatch.setattr(c, "ASR_MIN_SPEAKERS", "2")
    monkeypatch.setattr(c, "ASR_MAX_SPEAKERS", "4")
    with app.app_context():
        resolved = resolve_transcription_params(None, {}, tags=[], folder=None, owner=None)
        assert (resolved["min_speakers"], resolved["max_speakers"]) == (2, 4)
        resolved = resolve_transcription_params(None, {"min_speakers": 3}, tags=[], folder=None, owner=None)
        assert (resolved["min_speakers"], resolved["max_speakers"]) == (3, 4)
