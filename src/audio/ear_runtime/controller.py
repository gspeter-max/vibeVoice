"""
Ear Runtime Controller
======================

The main audio capture and processing engine for Parakeet Flow.
The Ear is responsible for opening the microphone stream, performing
real-time VAD (Voice Activity Detection), and splitting speech into
logical chunks. It coordinates with the Brain via sockets to send
audio data and with the HUD to provide visual feedback to the user.
"""

import os
import sys
import socket
import threading
import time
import pyaudio
import numpy as np

from src import log
from src.streaming.session import (
    should_split_chunk_after_silence,
)
from src.audio.vad_segmenter import SileroVAD, SileroUtteranceGate
from src.audio.ear_runtime.analysis import (
    analyze_frequency_bands,
    boost_audio_chunk,
    get_rms as runtime_get_rms,
)
from src.audio.ear_runtime.menu import (
    TerminalMenu as RuntimeTerminalMenu,
)
from src.audio.ear_runtime.devices import resolve_input_device_index
from src.audio.ear_runtime.system_audio import (
    load_start_sound,
    play_start_sound,
)
from src.ipc.client import (
    close_raw_audio_stream_and_forget,
    open_checked_raw_audio_stream_to_brain,
    send_message_to_brain,
    send_raw_audio_stream_chunk_or_close,
)
from src.ipc.protocol import (
    format_audio_chunk_message,
    format_session_commit_message,
    format_session_event_message,
)
from src.streaming.capture_session import CaptureSession
from src.ui.hud_client import start_hud_command_thread, start_volume_sender_thread
from src.utils.settings import settings

try:
    from pynput import keyboard, mouse
except Exception:  # pragma: no cover
    class _FallbackKey:
        cmd_r = "cmd_r"
        esc = "esc"

    class _FallbackKeyboardListener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            """Match pynput's listener API without doing any work."""
            return None

    class _FallbackKeyboardModule:
        Key = _FallbackKey()
        Listener = _FallbackKeyboardListener

    class _FallbackMouseButton:
        left = "left"
        right = "right"

    class _FallbackMouseListener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            """Match pynput's listener API without doing any work."""
            return None

    class _FallbackMouseModule:
        Button = _FallbackMouseButton()
        Listener = _FallbackMouseListener

    keyboard = _FallbackKeyboardModule()
    mouse = _FallbackMouseModule()

# Initialize sounds
load_start_sound()


def _is_right_cmd(key) -> bool:
    """Return whether a keyboard event matches the Right Command hotkey."""
    return (
        key == keyboard.Key.cmd_r
        or getattr(key, "name", None) == "cmd_r"
        or getattr(key, "vk", None) == settings.right_cmd_vk
    )

TerminalMenu = RuntimeTerminalMenu


# ── Main ear class (STREAMING) ─────────────────────────────────────────────────
class Ear:
    """
    The main audio capture and processing engine for Parakeet Flow.
    The Ear is responsible for opening the microphone stream, performing
    real-time VAD (Voice Activity Detection), and splitting speech into
    logical chunks. It coordinates with the Brain via sockets to send
    audio data and with the HUD to provide visual feedback to the user.
    """
    def __init__(self, input_device_index=None, pyaudio_lib=None):
        self.pyaudio_library_for_capturing_audio = pyaudio_lib or pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self._lock = threading.Lock()
        self.last_rms = 0.0
        self.gain_multiplier = 1.2  # Increased from 1.1 to fix quiet mic issues
        self._total_frames = 0
        self.last_frequency_bands = {"bass": 0.33, "mid": 0.33, "treble": 0.34}
        self.input_device_index = resolve_input_device_index(
            self.pyaudio_library_for_capturing_audio,
            input_device_index,
        )

        self.active_mic_name = (
            self.pyaudio_library_for_capturing_audio
            .get_device_info_by_index(self.input_device_index)
            .get("name")
        )

        self._cmd_press_time = 0.0
        self._toggle_active = False
        self._brain_sock = None
        self._brain_sock_lock = threading.Lock()
        self._telemetry_enabled = (
            os.environ.get("STREAMING_TELEMETRY_ENABLED", "0").strip() == "1"
        )

        self._chunk_speech_logged = False
        self._silence_pending_logged = False
        self._vad_state_log_time = 0.0
        self._recording_level_log_time = 0.0
        self._vad_no_speech_warned = False
        # Session ID is generated ONCE when the app launches — it never changes.
        # _recording_index tracks how many times the user has pressed the record button.
        self._capture_session = CaptureSession(
            sample_rate=settings.rate,
            overlap_seconds=settings.overlap_seconds,
        )
        self.current_model = "parakeet-tdt-0.6b-v3"  # Default model

        # ★ VAD: buffer full utterances locally before sending to Brain
        try:
            self._vad_engine = SileroVAD(settings.vad_model_path)
            log.info("[Ear] Silero VAD loaded ✓")
        except Exception as e:
            self._vad_engine = None
            log.info(f"[Ear] ⚠️ Silero VAD load failed: {e} — using buffer-only fallback")

        self._utterance_gate = SileroUtteranceGate(
            self._vad_engine,
            voice_threshold=settings.vad_score_threshold,
            silence_timeout_s=settings.silence_timeout_seconds,
            energy_threshold=settings.vad_energy_threshold,
            energy_ratio=settings.vad_energy_ratio,
        )
        log.info(
            "[Ear] VAD config: "
            f"threshold={settings.vad_score_threshold:.2f}, "
            f"silence_timeout={settings.silence_timeout_seconds:.2f}s, "
            f"energy_threshold={settings.vad_energy_threshold:.3f}, "
            f"energy_ratio={settings.vad_energy_ratio:.2f}"
        )

        # ★ ALWAYS LISTENING MODE: Open stream once at startup
        self.stream = None
        self._open_mic_stream()
        log.info(f"[Ear] Mic selected: {self.active_mic_name} ✓")

    def _begin_recording_session(self):
        """
        Initializes a new streaming session with a unique ID.
        This function resets the chunk counters and records the start time,
        then sends a 'session_started' event to the Brain. This ensures
        that both the Ear and the Brain are synchronized and ready to
        process a series of related audio chunks as a single conversation.
        """
        # session_id was already created in __init__; just reset the per-recording counters.
        self._capture_session.begin_recording(time.time())
        self._send_session_event_to_brain(
            "session_started",
            {"recording_mode": settings.recording_mode},
        )

    def _send_session_event_to_brain(self, event_type: str, fields: dict | None = None) -> bool:
        """
        Transmits a telemetry event from the Ear to the Brain's server.
        It packs the event type and optional data fields into a JSON
        message and sends it over the Unix domain socket. This allows
        the Brain to keep track of microphone levels and VAD state
        changes that happen on the Ear's side of the application.
        """
        if not self._telemetry_enabled or not self._capture_session.current_session_id:
            return False

        payload = {"type": event_type}
        if fields:
            payload.update(fields)
        message_bytes = format_session_event_message(
            self._capture_session.current_session_id,
            self._capture_session.current_recording_index,
            payload,
        )
        sent = send_message_to_brain(
            message_bytes,
            timeout_seconds=5.0,
            socket_path=settings.socket_path,
            socket_factory=socket.socket,
        )
        if not sent:
            log.info(f"[Ear] ❌ Failed to send telemetry event '{event_type}'")
        return sent

    def _send_audio_chunk_to_brain(self, utterance_bytes: bytes) -> bool:
        """
        Sends a single processed audio chunk to the Brain for transcription.
        The chunk is prefixed with a session header and a sequence number
        so the Brain can correctly stitch it back together with other chunks.
        It also records a 'chunk_sent_to_brain' event to the telemetry log
        to help debug any network latency or data loss between processes.
        """
        if not utterance_bytes or not self._capture_session.current_session_id:
            return False

        session_id = self._capture_session.current_session_id
        recording_index = self._capture_session.current_recording_index
        seq = self._capture_session.mark_chunk_sent()
        message_bytes = format_audio_chunk_message(
            session_id,
            recording_index,
            seq,
            utterance_bytes,
        )
        sent = send_message_to_brain(
            message_bytes,
            timeout_seconds=5.0,
            socket_path=settings.socket_path,
            socket_factory=socket.socket,
        )
        if sent:
            self._send_session_event_to_brain(
                "chunk_sent_to_brain",
                {
                    "chunk_index": seq,
                    "audio_bytes": len(utterance_bytes),
                },
            )
            return True
        log.error("❌ Failed to send chunk")
        return False

    def _commit_recording_session(self) -> bool:
        """
        Signals to the Brain that the current recording session is finished.
        This command is sent when the user releases the recording hotkey.
        It tells the Brain that no more audio chunks will be arriving for
        this session ID, allowing the Brain to finalize the transcription
        and paste the text as soon as all pending chunks are processed.
        """
        if not self._capture_session.current_session_id:
            return False

        session_id = self._capture_session.current_session_id
        recording_index = self._capture_session.current_recording_index
        message_bytes = format_session_commit_message(
            session_id,
            recording_index,
        )
        sent = send_message_to_brain(
            message_bytes,
            timeout_seconds=5.0,
            socket_path=settings.socket_path,
            socket_factory=socket.socket,
        )
        if sent:
            log.info("✅ Session committed", session=session_id[:8], recording=recording_index)
            self._capture_session.mark_recording_committed()
            return True
        log.error("❌ Failed to commit session")
        return False

    def _reset_chunk_tracking(self):
        """
        Resets internal flags and counters for a new audio chunk.
        When a chunk is finalized or split, we need to clear the state
        that tracks whether we've already logged speech or warned about
        silence for that specific segment. This ensures that log messages
        and telemetry events are correctly associated with each new
        part of the user's ongoing speech.
        """
        self._chunk_speech_logged = False
        self._silence_pending_logged = False
        self._vad_no_speech_warned = False

    def _flush_current_chunk(self, *, stop_session: bool) -> bool:
        """
        Finalizes the current utterance and sends it to the Brain.
        This function is called either when silence is detected or when
        the user stops recording. It collects all buffered audio from the
        VAD, applies gain and overlap, and then transmits the final result.
        It also handles the session commit command if this was the last chunk
        of the recording, effectively closing the loop on that session.
        """
        now = time.time()
        silence_elapsed = self._utterance_gate.silence_elapsed(now)

        with self._lock:
            if not self.is_recording:
                return False
            total = self._total_frames
            if stop_session:
                self.is_recording = False
            self._total_frames = 0
            self.last_rms = 0.0
            self._reset_chunk_tracking()

        utterance = self._utterance_gate.flush()
        if not utterance:
            if stop_session:
                self._capture_session.mark_recording_stopped()
                log.info("[Ear] 🔇 No speech captured; stopping recording")
                # Always commit so the Brain closes out this recording slot
                self._commit_recording_session()
            return False

        utterance_for_brain = boost_audio_chunk(utterance, self.gain_multiplier)
        last_chunk_tail_bytes = self._capture_session.last_chunk_tail_bytes
        utterance_for_brain = self._capture_session.prepare_chunk_for_send(
            utterance_for_brain,
            stop_session=stop_session,
            silence_seconds=silence_elapsed if not stop_session else 0.0
        )
        overlap_seconds_added = len(last_chunk_tail_bytes) / 2.0 / settings.rate
        chunk_age_seconds = self._capture_session.current_chunk_age_seconds(now)

        duration = (total * settings.chunk) / settings.rate
        if stop_session:
            log.info(f"\r\n⏹️  Streamed {duration:.1f}s ({total} chunks) — Brain transcribing...\n")
            start_hud_command_thread("process", socket_factory=socket.socket)
        else:
            log.info(
                f"\r[Ear] ✂️  Silence boundary hit ({silence_elapsed:.2f}s) — sending chunk "
                f"{duration:.1f}s ({total} chunks)"
            )

        sent = self._send_audio_chunk_to_brain(utterance_for_brain)
        if sent:
            self._send_session_event_to_brain(
                "silence_threshold_hit" if not stop_session else "session_stopped",
                {
                    "chunk_index": self._capture_session.current_chunk_sequence_number - 1,
                    "chunk_age_seconds": round(chunk_age_seconds, 2),
                    "silence_elapsed_seconds": round(silence_elapsed, 2),
                    "split_reason": "silence_threshold_hit" if not stop_session else "session_stop",
                    "overlap_seconds_added": round(overlap_seconds_added, 4),
                    "audio_bytes": len(utterance_for_brain),
                },
            )
        if stop_session:
            self._capture_session.mark_recording_stopped()
            self._commit_recording_session()
        else:
            self._capture_session.mark_nonfinal_chunk_sent()
        return sent

    def _stop_no_streaming(self):
        """
        Stops a non-streaming recording session and closes the Brain socket.
        In legacy non-streaming mode, we don't split audio into chunks.
        Instead, we wait until the user finishes and then signal the Brain
        to process the entire buffer at once. This function manages that
        transition by updating the recording state, notifying the HUD,
        and cleanly shutting down the background socket connection.
        """
        with self._lock:
            if not self.is_recording:
                return
            self.is_recording = False
            total = self._total_frames
            self._total_frames = 0
            self.last_rms = 0.0

        duration = (total * settings.chunk) / settings.rate
        log.info(f"\r\n⏹️  Streamed {duration:.1f}s ({total} chunks) — Brain transcribing...\n")
        start_hud_command_thread("process", socket_factory=socket.socket)
        with self._brain_sock_lock:
            raw_stream_socket = self._brain_sock
            self._brain_sock = None
        threading.Thread(
            target=close_raw_audio_stream_and_forget,
            args=(raw_stream_socket,),
            daemon=True,
        ).start()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle one microphone callback and route it to the active mode."""
        del frame_count, time_info, status

        with self._lock:
            if not self.is_recording:
                return (None, pyaudio.paContinue)

            chunk_bytes = boost_audio_chunk(in_data, self.gain_multiplier)
            boosted = np.frombuffer(chunk_bytes, dtype=np.int16)
            self.last_rms = runtime_get_rms(chunk_bytes)
            self._total_frames += 1

            # ★ FREQUENCY ANALYSIS: Extract bass/mid/treble bands using FFT
            self.last_frequency_bands = analyze_frequency_bands(
                boosted,
                sample_rate=settings.rate,
            )

            if settings.is_no_streaming_mode:
                # Raw mode: send bytes straight to Brain and wait for socket close.
                with self._brain_sock_lock:
                    self._brain_sock = send_raw_audio_stream_chunk_or_close(
                        self._brain_sock,
                        chunk_bytes,
                    )
                    raw_stream_socket_alive = self._brain_sock is not None
                if not raw_stream_socket_alive:
                    log.info("\r⚠️  Brain disconnected — will transcribe on release\n")
            else:
                # Silence mode: keep the utterance locally until silence splits it.
                now = time.time()

                speech_now = self._utterance_gate.push(
                    audio_chunk=in_data,
                    now=now,
                    analysis_chunk=chunk_bytes
                )

                if speech_now and not self._chunk_speech_logged:
                    self._chunk_speech_logged = True
                    self._silence_pending_logged = False
                    log.info("[Ear] 🗣️  VAD speech detected")

                if now - self._vad_state_log_time >= settings.vad_status_log_interval:
                    try:
                        score = self._utterance_gate.last_score()
                        energy = self._utterance_gate.last_energy()
                        dynamic_threshold = self._utterance_gate.last_dynamic_threshold()
                        started = self._utterance_gate.has_speech_started()
                        silence_elapsed = (
                            self._utterance_gate.silence_elapsed(now)
                            if started
                            else 0.0
                        )
                        log.debug(
                            "[Ear] 🔎 "
                            f"VAD score={score:.3f} "
                            f"threshold={settings.vad_score_threshold:.2f} "
                            f"started={started} silence={silence_elapsed:.2f}s "
                            f"rms={self.last_rms:.4f} "
                            f"energy={energy:.4f} energy_threshold={dynamic_threshold:.4f}",
                        )
                    except Exception:
                        pass
                    self._vad_state_log_time = now

        return (None, pyaudio.paContinue)

    def _open_mic_stream(self):
        """Open a FRESH mic stream each recording session."""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass

        self.stream = self.pyaudio_library_for_capturing_audio.open(
            format=settings.audio_format,
            channels=settings.channels,
            rate=settings.rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=settings.chunk,
            stream_callback=self._audio_callback,
        )
        log.info("[Ear] 🎤 Mic stream opened")

    def _close_mic_stream(self):
        """
        Shuts down the active PyAudio microphone stream.
        This function stops the audio input and releases the hardware
        resources. It is called when the application is closing to ensure
        that the microphone isn't left in an 'active' state by the OS,
        which could prevent other applications from using it or cause
        unnecessary battery drain on mobile devices like MacBooks.
        """
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _start_recording_state(self, *, from_hold: bool) -> None:
        """
        Resets the internal state and prepares the Ear for a new recording.
        It clears all volume and frame counters, sets the recording flag,
        and determines if this recording was triggered by a sustained hold
        or a single keypress. In streaming mode, it also resets the VAD
        engine and starts a fresh telemetry session to track the new recording.
        """
        del from_hold
        play_start_sound()

        with self._lock:
            self.is_recording = True
            self.last_rms = 0.0
            self._total_frames = 0
            self._reset_chunk_tracking()
            self._recording_level_log_time = 0.0

        self._capture_session.clear_overlap_tail()
        if settings.is_silence_streaming_mode:
            self._utterance_gate.reset()
            self._begin_recording_session()

    def on_press(self, key):
        """
        Keyboard event handler for when a key is pressed down.
        It specifically listens for the Right Command key to trigger the
        recording state. Depending on whether the app is in streaming or
        non-streaming mode, it will either start buffering chunks locally
        or open a persistent socket to the Brain. It also notifies the HUD
        to display the 'listening' state to provide visual confirmation.
        """
        if not _is_right_cmd(key):
            return

        log.debug("[Ear] Right CMD pressed")

        if self._toggle_active:
            self._toggle_active = False
            self._stop_and_send(stop_session=True)
            return

        with self._lock:
            if self.is_recording:
                log.info("[Ear] ⚠️ Already recording, ignoring press")
                return

        if settings.is_no_streaming_mode:
            log.debug("[Ear] About to open brain stream")
            raw_stream_socket = open_checked_raw_audio_stream_to_brain(
                timeout_seconds=5.0,
                socket_path=settings.socket_path,
                socket_factory=socket.socket,
            )
            if raw_stream_socket is None:
                log.info("[Ear] ❌ Failed to open brain stream, aborting")
                return
            with self._brain_sock_lock:
                self._brain_sock = raw_stream_socket

        self._start_recording_state(from_hold=False)

        self._cmd_press_time = time.time()
        log.info("\r\n" + "─" * 50)
        log.info(f"\r🎙️  RECORDING ({self.active_mic_name})")

        start_hud_command_thread("listen", socket_factory=socket.socket)
        start_volume_sender_thread(
            self,
            volume_port=settings.vol_port,
            socket_factory=socket.socket,
        )

    def on_release(self, key):
        """
        Keyboard event handler for when a key is released.
        If the Right Command key is released after being held for more
        than a short threshold, it stops the recording and signals the
        Brain to finalize. If the release happens very quickly, the app
        enters 'toggle mode', where recording continues until the key
        is pressed again, allowing for hands-free dictation when needed.
        """
        if not _is_right_cmd(key):
            return

        if self._toggle_active:
            return

        with self._lock:
            if not self.is_recording:
                return

        press_duration = time.time() - self._cmd_press_time
        if press_duration >= settings.recording_button_hold_threshold:
            log.info("\r[Ear] ⏹️  Right CMD released - finalizing now")
            self._stop_and_send(stop_session=True)
        else:
            self._toggle_active = True
            log.info("\r\n⏸️  Toggle mode — tap Right CMD again to stop")

    def _stop_and_send(self, *, stop_session: bool = True):
        """
        Unified method to stop recording and transmit the final data.
        It routes the stop command to either the streaming or non-streaming
        handler depending on the current configuration. This ensures that
        all recording logic follows the same cleanup path, whether it was
        triggered by a keyboard release, a mouse release, or a toggle click.
        """
        if settings.is_no_streaming_mode:
            self._stop_no_streaming()
            return
        self._flush_current_chunk(stop_session=stop_session)

    def record_loop(self, input_trigger=None):
        """
        Main background loop that manages the active recording state.

        Args:
            input_trigger: The InputTrigger instance from hotkeys.py. When provided,
                           each tick calls input_trigger.check_mouse_hold_threshold()
                           to handle right-mouse-button hold-to-record activation.
        """
        while True:
            time.sleep(0.05)
            self._record_loop_tick(input_trigger=input_trigger)

    def _record_loop_tick(self, input_trigger=None):
        """
        Internal tick function called by the record loop to update state.

        Delegates mouse hold-to-record checking to InputTrigger, which is the
        single source of truth for mouse button state. On every tick, if an
        input_trigger is provided, it polls check_mouse_hold_threshold() to
        see if the right mouse button has been held for >= 1 second.

        Args:
            input_trigger: Optional InputTrigger instance. When provided,
                           mouse hold-to-record is active.
        """
        with self._lock:
            recording = self.is_recording
            rms = self.last_rms

        # Delegate mouse hold check to InputTrigger — it owns all mouse state.
        if input_trigger is not None:
            input_trigger.check_mouse_hold_threshold()

        # Display volume meter when recording
        if recording:
            now = time.time()
            if now - self._recording_level_log_time >= settings.recording_level_log_interval:
                meter_width = 30
                level = min(int(rms * 300), meter_width)
                meter = "█" * level + "░" * (meter_width - level)
                # Use \r to keep the meter on a single line
                sys.stdout.write(f"\r  Voice Level: [{meter}] ")
                sys.stdout.flush()
                self._recording_level_log_time = now

            if settings.is_silence_streaming_mode:
                # --- Nemotron Fixed Time Gap Logic ---
                if "nemotron" in self.current_model.lower():
                    time_since_last_chunk = self._capture_session.current_chunk_age_seconds(now)
                    if time_since_last_chunk >= 1.12:
                        # Log removed for Zen mode
                        self._stop_and_send(stop_session=False)
                    return

                # Silence mode watches for speech ending
                if self._utterance_gate.has_speech_started() and not self._silence_pending_logged:
                    silence_elapsed = self._utterance_gate.silence_elapsed(now)
                    if silence_elapsed > 0.0:
                        self._silence_pending_logged = True

                silence_elapsed = (
                    self._utterance_gate.silence_elapsed(now)
                    if self._utterance_gate.has_speech_started()
                    else self._utterance_gate.finalize_elapsed(now)
                )
                split_decision = should_split_chunk_after_silence(
                    chunk_started_at_seconds=self._capture_session.chunk_started_at_seconds,
                    now_seconds=now,
                    minimum_chunk_age_before_silence_split_seconds=(
                        settings.minimum_chunk_age_before_silence_split_seconds
                    ),
                    utterance_gate_should_finalize_now=self._utterance_gate.should_finalize(now),
                    silence_duration_seconds=silence_elapsed,
                )
                if split_decision.should_split_now:
                    # Log removed for Zen mode
                    self._stop_and_send(stop_session=False)

    def cleanup(self):
        """
        Performs a clean shutdown of all Ear resources.
        """
        with self._brain_sock_lock:
            raw_stream_socket = self._brain_sock
            self._brain_sock = None
        close_raw_audio_stream_and_forget(raw_stream_socket)
        self._close_mic_stream()
        self.pyaudio_library_for_capturing_audio.terminate()
