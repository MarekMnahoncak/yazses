"""Tests for the SLM model cache, without a network or the user's real cache directory.

`urlretrieve` is a mock with fake behaviour: it writes a file to the real destination in
`tmp_path`, so the tests check what ends up on disk as well as what was called.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, MagicMock
from urllib.error import URLError

import pytest

from yazses.commands import model_manager
from yazses.commands.model_manager import ModelInfo, download_model, local_path

_TINY = ModelInfo(
    id="tiny",
    filename="tiny.gguf",
    url="https://example.invalid/tiny.gguf",
    size_mb=1,
    description="A model that only exists in this test",
)


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """Point the module at a one-model registry and a private cache; return the cache dir."""
    directory = tmp_path / "models"
    monkeypatch.setattr(model_manager, "_CACHE_BASE", directory)
    monkeypatch.setattr(model_manager, "REGISTRY", {_TINY.id: _TINY})
    return directory


@pytest.fixture
def urlretrieve(monkeypatch):
    """Replace the module's view of urllib, so no test can reach the network."""
    mock = MagicMock(name="urlretrieve")
    monkeypatch.setattr(model_manager, "urllib", SimpleNamespace(request=SimpleNamespace(urlretrieve=mock)))
    return mock


def test_an_unknown_model_is_refused_and_the_error_names_the_known_ones(cache, urlretrieve):
    with pytest.raises(ValueError, match=r"Unknown model 'nope'\. Known models: tiny"):
        download_model("nope")

    assert local_path("nope") is None
    urlretrieve.assert_not_called()


def test_a_model_already_in_the_cache_is_not_downloaded_again(cache, urlretrieve):
    cache.mkdir()
    existing = cache / _TINY.filename
    existing.write_bytes(b"already here")

    assert local_path(_TINY.id) == existing
    assert download_model(_TINY.id) == existing

    urlretrieve.assert_not_called()
    assert existing.read_bytes() == b"already here"


@pytest.mark.parametrize("show_progress", [True, False], ids=["with-progress", "quiet"])
def test_a_fresh_download_lands_in_the_cache_and_prints_only_when_asked(cache, urlretrieve, capsys, show_progress):
    def write_the_file(_url, dest, reporthook):
        reporthook(1, 8192, -1)  # a server that sends no Content-Length
        reporthook(1, 8192, 16384)
        dest.write_bytes(b"gguf")

    urlretrieve.side_effect = write_the_file
    assert local_path(_TINY.id) is None

    path = download_model(_TINY.id, show_progress=show_progress)

    assert path == cache / _TINY.filename
    assert path.read_bytes() == b"gguf"
    assert local_path(_TINY.id) == path
    urlretrieve.assert_called_once_with(_TINY.url, path, reporthook=ANY)
    out = capsys.readouterr().out
    if show_progress:
        assert "Downloading tiny (1 MB)" in out
        assert " 50%" in out
        assert "100%" in out
        assert f"Saved to {path}" in out
    else:
        assert out == ""


def test_a_failed_download_leaves_no_partial_file_behind(cache, urlretrieve):
    def fail_halfway(_url, dest, reporthook):
        dest.write_bytes(b"half a mod")
        raise URLError("connection reset")

    urlretrieve.side_effect = fail_halfway

    with pytest.raises(URLError, match="connection reset"):
        download_model(_TINY.id, show_progress=False)

    assert not (cache / _TINY.filename).exists()
    assert local_path(_TINY.id) is None
