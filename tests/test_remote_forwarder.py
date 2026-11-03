"""Tests for the SSH tunnel manager, without ssh, a network or real time passing.

The module's collaborators are swapped for test doubles:
  - FakeProcess: a working stand-in for the ssh process
  - FakeThread: records the monitor thread instead of starting it
  - which/sleep stubs: canned answers, so ssh "is installed" and nothing waits
  - Popen mock: records the command the forwarder asked to run
"""
from __future__ import annotations

import io
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yazses.remote import forwarder as forwarder_module
from yazses.remote.forwarder import RemoteForwarder


class FakeProcess:
    """Behaves like a running Popen, tracking signals instead of owning an OS process."""

    def __init__(self, returncode=None, stderr=b"", ignores_terminate=False):
        self.pid = 4242
        self.returncode = returncode
        self.stderr = io.BytesIO(stderr)
        self.ignores_terminate = ignores_terminate
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignores_terminate:
            self.returncode = -15

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("ssh", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeThread:
    """Records how the monitor thread was set up; `start` never runs the target."""

    def __init__(self, target, daemon):
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


@pytest.fixture
def popen(monkeypatch):
    """Replace everything `connect` reaches for, and return the Popen mock.

    The module attributes are replaced rather than `subprocess.Popen` itself, so nothing
    else in the process sees the doubles.
    """
    mock = MagicMock(name="Popen", return_value=FakeProcess())
    monkeypatch.setattr(
        forwarder_module,
        "subprocess",
        SimpleNamespace(Popen=mock, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.setattr(forwarder_module, "shutil", SimpleNamespace(which=lambda name: f"/usr/bin/{name}"))
    monkeypatch.setattr(forwarder_module, "time", SimpleNamespace(sleep=lambda _seconds: None))
    monkeypatch.setattr(forwarder_module, "threading", SimpleNamespace(Thread=FakeThread))
    return mock


@pytest.mark.parametrize(
    ("options", "expected_options"),
    [
        ({}, []),
        ({"port": 2222, "key_file": "/keys/id_ed25519"}, ["-i", "/keys/id_ed25519", "-p", "2222"]),
    ],
    ids=["defaults", "key-and-port"],
)
def test_connect_runs_ssh_with_the_reverse_tunnel_and_only_the_requested_options(popen, options, expected_options):
    RemoteForwarder().connect("user@host", **options)

    popen.assert_called_once_with(
        [
            "ssh", "-o", "ExitOnForwardFailure=yes", "-o", "StrictHostKeyChecking=accept-new",
            *expected_options,
            "-R", "9875:127.0.0.1:9875",
            "user@host",
            "yazses-agent", "--listen", "9875",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_connect_marks_the_tunnel_up_and_starts_a_daemon_monitor(popen):
    forwarder = RemoteForwarder()

    forwarder.connect("user@host")

    assert forwarder.is_connected()
    thread = forwarder._monitor_thread
    assert thread.started and thread.daemon
    assert thread.target == forwarder._monitor


def test_connect_reports_what_ssh_said_when_the_tunnel_dies_at_startup(popen):
    popen.return_value = FakeProcess(returncode=255, stderr=b"Permission denied (publickey).\n")
    forwarder = RemoteForwarder()

    with pytest.raises(RuntimeError, match=r"SSH tunnel failed to start: Permission denied \(publickey\)\.$"):
        forwarder.connect("user@host")

    assert not forwarder.is_connected()
    assert forwarder._monitor_thread is None


def test_connect_while_connected_tears_down_the_old_tunnel_first(popen):
    first, second = FakeProcess(), FakeProcess()
    popen.side_effect = [first, second]
    forwarder = RemoteForwarder()
    forwarder.connect("old@host")

    forwarder.connect("new@host")

    assert first.terminated
    assert not second.terminated
    assert forwarder._process is second
    assert forwarder.is_connected()


@pytest.mark.parametrize("ignores_terminate", [False, True], ids=["exits-politely", "ignores-terminate"])
def test_disconnect_terminates_ssh_and_kills_it_only_if_it_will_not_exit(popen, ignores_terminate):
    process = FakeProcess(ignores_terminate=ignores_terminate)
    popen.return_value = process
    forwarder = RemoteForwarder()
    forwarder.connect("user@host")

    forwarder.disconnect()

    assert process.terminated
    assert process.killed is ignores_terminate
    assert not forwarder.is_connected()


def test_monitor_notices_when_ssh_exits_on_its_own(monkeypatch):
    process = FakeProcess()

    def ssh_dies_during_the_sleep(_seconds):
        process.returncode = 255

    monkeypatch.setattr(forwarder_module, "time", SimpleNamespace(sleep=ssh_dies_during_the_sleep))
    forwarder = RemoteForwarder()
    forwarder._process = process
    forwarder._connected = True
    assert forwarder.is_connected()

    forwarder._monitor()

    assert forwarder._connected is False
    assert not forwarder.is_connected()
