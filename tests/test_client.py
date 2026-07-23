import socket
from unittest.mock import MagicMock, patch

import pytest

from src.ipc.client import (
    SocketConfig,
    commit_stop,
    create_socket,
    send_audio,
    send_event,
    send_message,
)
from src.audio.capture_session import CaptureSession


class FakeSocket:
    def __init__(self):
        self.connected_address = None
        self.timeout = None
        self.sent_payloads = []
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        self.connected_address = address

    def sendall(self, payload):
        self.sent_payloads.append(payload)

    def close(self):
        self.closed = True


def test_socket_config_defaults():
    cfg = SocketConfig()
    assert cfg.family == socket.AF_UNIX
    assert cfg.socket_type == socket.SOCK_STREAM
    assert cfg.timeout is None


def test_create_socket_connects_and_closes_in_finally():
    fake_sock = FakeSocket()
    cfg = SocketConfig(address="/tmp/test.sock", timeout=2.0)

    with patch("src.ipc.client.socket.socket", return_value=fake_sock):
        with create_socket(cfg) as sock:
            assert sock is fake_sock
            assert fake_sock.connected_address == "/tmp/test.sock"
            assert fake_sock.timeout == 2.0
            assert fake_sock.closed is False

        assert fake_sock.closed is True


def test_create_socket_closes_on_exception():
    fake_sock = FakeSocket()
    cfg = SocketConfig(address="/tmp/test.sock")

    with patch("src.ipc.client.socket.socket", return_value=fake_sock):
        with pytest.raises(RuntimeError):
            with create_socket(cfg):
                raise RuntimeError("Decode error")

        assert fake_sock.closed is True


def test_create_socket_raises_file_not_found_error():
    cfg = SocketConfig(address="/tmp/nonexistent.sock")
    with patch("src.ipc.client.socket.socket") as mock_sock_cls:
        mock_sock_inst = mock_sock_cls.return_value
        mock_sock_inst.connect.side_effect = FileNotFoundError("No socket file")

        with pytest.raises(FileNotFoundError, match="Socket connection failed"):
            with create_socket(cfg):
                pass


def test_send_message_returns_false_for_empty_message():
    assert send_message(b"") is False


def test_send_message_success():
    fake_sock = FakeSocket()
    cfg = SocketConfig(address="/tmp/test.sock")

    with patch("src.ipc.client.socket.socket", return_value=fake_sock):
        assert send_message(b"hello world", cfg) is True
        assert fake_sock.sent_payloads == [b"hello world"]


def test_send_message_returns_false_on_oserror():
    cfg = SocketConfig(address="/tmp/test.sock")
    with patch("src.ipc.client.socket.socket") as mock_sock_cls:
        mock_sock_inst = mock_sock_cls.return_value
        mock_sock_inst.connect.side_effect = OSError("Connection failed")

        assert send_message(b"test payload", cfg) is False


def test_commit_stop_commits_session_on_success():
    session = CaptureSession(16000, 0.5, session_id="session123", rec_idx=0)
    cfg = SocketConfig(address="/tmp/test.sock")
    fake_sock = FakeSocket()

    with patch("src.ipc.client.socket.socket", return_value=fake_sock):
        assert commit_stop(session, cfg) is True
        assert session.rec_idx == 1
        assert len(fake_sock.sent_payloads) == 1
        assert b"CMD_SESSION_COMMIT" in fake_sock.sent_payloads[0]


def test_commit_stop_returns_false_if_no_session_id():
    session = CaptureSession(16000, 0.5, session_id="", rec_idx=0)
    assert commit_stop(session) is False


def test_send_event_sends_telemetry_when_enabled():
    session = CaptureSession(16000, 0.5, session_id="session123", rec_idx=0)
    fake_sock = FakeSocket()

    with patch("src.ipc.client.socket.socket", return_value=fake_sock):
        success = send_event(
            session=session,
            telemetry_enabled=True,
            event_type="test_event",
            fields={"key": "value"},
        )
        assert success is True
        assert len(fake_sock.sent_payloads) == 1
        assert b"test_event" in fake_sock.sent_payloads[0]


def test_send_event_returns_false_when_telemetry_disabled():
    session = CaptureSession(16000, 0.5, session_id="session123", rec_idx=0)
    assert send_event(session, telemetry_enabled=False, event_type="test") is False


def test_send_audio_sends_payload_and_emits_event():
    session = CaptureSession(16000, 0.5, session_id="session123", rec_idx=0)
    fake_sock = FakeSocket()

    with patch("src.ipc.client.socket.socket", return_value=fake_sock):
        success = send_audio(
            session=session,
            audio_bytes=b"\x00\x01\x02\x03",
            telemetry_enabled=True,
        )
        assert success is True
        assert len(fake_sock.sent_payloads) == 2
        assert b"CMD_AUDIO_CHUNK" in fake_sock.sent_payloads[0]
        assert b"chunk_sent_to_brain" in fake_sock.sent_payloads[1]
