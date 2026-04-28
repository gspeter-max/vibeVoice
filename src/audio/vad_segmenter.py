"""
Simple helpers for voice activity detection.

This file does two jobs:

1. Run the Silero VAD model on small audio frames.
2. Keep track of one spoken utterance so the caller can know:
   - when speech started
   - when silence has lasted long enough
   - when it is time to flush the saved audio

This file works with 16 kHz mono PCM16 audio.
It does not do speech-to-text.
It only decides whether audio looks like speech or silence.
"""

from __future__ import annotations

import numpy as np


class SileroVAD:
    """
    High-performance wrapper for the Silero Voice Activity Detection model.
    This class manages the loading and execution of the ONNX-based AI model,
    handling different model versions (V3 and V5) and their specific state
    buffer requirements. It is optimized to run on CPU and provides real-time
    classification of small audio frames, telling the system whether the
    current microphone input contains human speech or just background noise.
    """

    def __init__(self, model_path: str):
        """
        Initializes the VAD engine by loading the ONNX model from the disk.
        It configures the ONNX runtime with single-threaded execution to
        minimize CPU overhead while still maintaining fast inference times.
        The constructor also detects the model version and allocates the
        necessary recurrent state buffers, ensuring that the model can
        maintain context across consecutive audio frames for higher accuracy.
        """
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

        state_input = next((inp for inp in self.session.get_inputs() if inp.name == "state"), None)
        if state_input:
            self._version = 5
            state_shape = [dim if isinstance(dim, int) else 1 for dim in state_input.shape] if state_input.shape else [2, 1, 128]
            self._state = np.zeros(state_shape, dtype=np.float32)
            # 64-sample rolling context window required by Silero V5 ONNX streaming.
            # Without it the model sees a cold start every frame and returns ~0.001.
            self._context = np.zeros((1, 64), dtype=np.float32)
        else:
            self._version = 3
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        """
        Clears the internal recurrent state of the AI model.
        This is called whenever a recording session ends or a silence
        boundary is hit, ensuring that the next piece of speech starts with
        a clean slate. Resetting the state prevents audio artifacts from
        one utterance from leaking into the next, which significantly
        improves the reliability of speech detection in noisy environments.
        """
        if self._version == 5:
            self._state = np.zeros_like(self._state)
            self._context = np.zeros_like(self._context)
        else:
            self._h = np.zeros_like(self._h)
            self._c = np.zeros_like(self._c)

    def is_speech(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Analyzes a single frame of audio and returns a speech probability score.
        It first normalizes the input frame to exactly 512 samples, padding or
        trimming as necessary to match the model's requirements. The function
        then executes the AI inference pass, updating its internal context and
        returning a score between 0.0 and 1.0. A higher score indicates a
        stronger confidence that the audio frame contains spoken human voice.
        """
        if len(audio_samples) == 0:
            return 0.0

        # The model expects a fixed-size frame, so we normalize the length here.
        if len(audio_samples) < 512:
            audio_samples = np.pad(audio_samples, (0, 512 - len(audio_samples)))
        elif len(audio_samples) > 512:
            audio_samples = audio_samples[:512]

        if self._version == 5:
            # Silero V5 ONNX streaming expects the current 512 samples preceded
            # by 64 samples of context from the previous frame. We maintain that
            # rolling window in self._context and prepend it here.
            audio_samples_2d = audio_samples.reshape(1, -1)
            input_with_context = np.concatenate([self._context, audio_samples_2d], axis=1)
            ort_inputs = {
                "input": input_with_context,
                "sr": np.array([sample_rate], dtype=np.int64),
                "state": self._state,
            }
            out, new_state = self.session.run(None, ort_inputs)
            self._state = new_state
            # Slide the context window forward: keep the last 64 samples.
            self._context = input_with_context[:, -64:].copy()
        else:
            ort_inputs = {
                "input": audio_samples.reshape(1, -1),
                "sr": np.array([sample_rate], dtype=np.int64),
                "h": self._h,
                "c": self._c,
            }
            out, h, c = self.session.run(None, ort_inputs)
            self._h, self._c = h, c

        return float(out[0][0])


class SileroUtteranceGate:
    """
    Logic engine for managing spoken utterances and silence detection.
    The UtteranceGate acts as a supervisor that watches the stream of
    individual audio frames and decides when a user has started talking
    and, more importantly, when they have stopped. It uses a combination
    of AI-based VAD scores and energy-based noise floor tracking to
    precisely identify speech boundaries in real-world environments.
    """

    def __init__(
        self,
        vad_engine,
        *,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        voice_threshold: float = 0.5,
        silence_timeout_s: float = 0.4,
        min_utterance_bytes: int = 8000,
        energy_threshold: float = 0.03,
        energy_ratio: float = 2.5,
    ):
        """
        Sets up the gate with specific sensitivity and timing parameters.
        The constructor initializes the internal buffers used for audio
        analysis and sets the thresholds for what counts as speech. It
        also establishes the 'silence timeout', which determines how long
        the system should wait after the last detected speech before
        finalizing the chunk and sending it to the Brain for transcription.
        """
        self.vad_engine = vad_engine
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.voice_threshold = voice_threshold
        self.silence_timeout_s = silence_timeout_s
        self.min_utterance_bytes = min_utterance_bytes
        self.energy_threshold = energy_threshold
        self.energy_ratio = energy_ratio
        self._buffer = bytearray()
        self._analysis_buffer = bytearray()
        self._raw_analysis_buffer = bytearray()
        self._speech_started = False
        self._last_voice_time = 0.0
        self._finalize_armed = False
        self._finalize_time = 0.0
        self._last_score = 0.0
        self._max_score = 0.0
        self._last_energy = 0.0
        self._last_dynamic_threshold = energy_threshold
        self._noise_floor = 0.0

    def reset(self):
        """
        Clears all buffered audio and resets the speech detection state.
        This is used to prepare for a completely new utterance, such as
        after a previous chunk has been successfully flushed. It ensures
        that counters, timers, and background noise estimates are reset,
        providing a clean starting point for identifying the next segment
         of user speech without interference from past recordings.
        """
        self._buffer.clear()
        self._analysis_buffer.clear()
        self._raw_analysis_buffer.clear()
        self._speech_started = False
        self._last_voice_time = 0.0
        self._finalize_armed = False
        self._finalize_time = 0.0
        self._last_score = 0.0
        self._max_score = 0.0
        self._last_energy = 0.0
        self._last_dynamic_threshold = self.energy_threshold
        self._noise_floor = 0.0
        if self.vad_engine is not None and hasattr(self.vad_engine, "reset"):
            self.vad_engine.reset()

    def arm_finalize(self, now: float) -> None:
        """
        Activates a countdown timer for closing the current utterance.
        This is called when the Ear process wants to stop recording, even
        if the VAD engine hasn't detected a long enough silence yet. It
        sets a timestamp that the gate uses to ensure the final few
        milliseconds of audio are processed and flushed before the
        application fully transitions back to its idle or hidden state.
        """
        self._finalize_armed = True
        self._finalize_time = now

    def has_speech_started(self) -> bool:
        """
        Returns True if human speech has been detected in the current chunk.
        This simple flag allows the caller to distinguish between a session
        that contains actual spoken content and one that was just background
        noise or an accidental button press. It is only set to True once
        the VAD score or energy level exceeds the project's configured
        sensitivity thresholds for a sustained period of time.
        """
        return self._speech_started

    def push(
            self, 
            audio_chunk: bytes, 
            now: float, 
            analysis_chunk : bytes | None = None
        ) -> bool:
        """
        Feeds new microphone data into the gate for real-time analysis.
        It buffers the incoming bytes and processes them in fixed-size
        frames, running each through the AI model and energy checks. The
        function also dynamically updates its internal 'noise floor' estimate,
        allowing the speech detection to automatically adapt to changing
        room environments like air conditioners or distant background chatter.
        """
        if not audio_chunk:
            return False
        self._raw_analysis_buffer.extend(audio_chunk)
        self._analysis_buffer.extend(analysis_chunk if analysis_chunk is not None else audio_chunk)
        # Keep the full utterance audio so it can be flushed later.
        self._buffer.extend(audio_chunk)

        frame_bytes = self.frame_samples * 2
        speech_detected = False
        while len(self._analysis_buffer) >= frame_bytes and len(self._raw_analysis_buffer) >= frame_bytes:
            # Read one analysis frame and remove it from the queue.
            frame_bytes_data = bytes(self._analysis_buffer[:frame_bytes])
            raw_frame_bytes = bytes(self._raw_analysis_buffer[:frame_bytes])
            del self._analysis_buffer[:frame_bytes]
            del self._raw_analysis_buffer[:frame_bytes]

            # Convert PCM16 bytes into normalized float audio for the model.
            audio = np.frombuffer(frame_bytes_data, dtype=np.int16).astype(np.float32) / 32768.0
            raw_audio_for_energy_detection = (
                np.frombuffer(raw_frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            score = 1.0 if self.vad_engine is None else self.vad_engine.is_speech(audio, sample_rate=self.sample_rate)
            frame_rms = float(np.sqrt(np.mean(raw_audio_for_energy_detection ** 2)))
            self._last_energy = frame_rms
            self._last_score = float(score)
            if score > self._max_score:
                self._max_score = float(score)

            # Build an energy threshold that can adapt to room noise.
            dynamic_threshold = max(self.energy_threshold, self._noise_floor * self.energy_ratio)
            self._last_dynamic_threshold = dynamic_threshold
            speech_by_energy = frame_rms >= dynamic_threshold

            if score > self.voice_threshold or speech_by_energy:
                # Any positive speech frame moves the utterance into speech state.
                speech_detected = True
                self._speech_started = True
                self._last_voice_time = now
            else:
                # Frames that are not speech help us slowly learn background noise.
                self._noise_floor = frame_rms if self._noise_floor == 0.0 else (0.95 * self._noise_floor) + (0.05 * frame_rms)

        return speech_detected

    def should_finalize(self, now: float) -> bool:
        """
        Determines if the current spoken segment is finished and ready to flush.
        A segment is considered complete if a significant period of silence
        has passed since the last speech frame was detected. It also checks
        the total length of the buffer to ensure we don't send extremely
        short, incomplete fragments to the AI, which helps maintain the
        overall transcription accuracy and context for the user.
        """
        if self._speech_started:
            # Do not finalize very tiny audio fragments after speech starts.
            if len(self._buffer) < self.min_utterance_bytes:
                return False
            return (now - self._last_voice_time) >= self.silence_timeout_s

        return (
            self._finalize_armed
            and (now - self._finalize_time) >= self.silence_timeout_s
        )

    def silence_elapsed(self, now: float) -> float:
        """
        Calculates the duration of the current silence in seconds.
        It measures the time gap between the current moment and the last
        timestamp where the system was confident that the user was still
        speaking. This value is used by the Ear's main loop to decide
        whether to split the recording into a new chunk or keep waiting
        for the user to continue their sentence or thought.
        """
        return now - self._last_voice_time

    def finalize_elapsed(self, now: float) -> float:
        """
        Returns the time since the 'finalize soon' timer was triggered.
        This provides a secondary safety check for ending a recording,
        ensuring that the application doesn't hang in a 'stopping' state
        indefinitely. If the finalize timer has been active for longer
        than the silence threshold, the gate assumes the recording is
        fully complete regardless of any lingering background noise.
        """
        return now - self._finalize_time

    def last_score(self) -> float:
        """
        Retrieves the AI confidence score from the most recent audio frame.
        This numeric value (0.0 to 1.0) gives developers and the UI real-time
        insight into whether the model currently 'hears' a voice. It is
        frequently used for internal debugging and for driving advanced
        HUD visualizations that might change color based on speech confidence.
        """
        return self._last_score

    def max_score(self) -> float:
        """
        Returns the highest confidence score achieved during this utterance.
        By tracking the peak speech score, the system can determine if a
        recording session ever actually contained high-confidence speech.
        This is a valuable metric for filtering out sessions that were
        triggered by accidental bumps to the microphone or short, non-speech
        sounds like a door closing or a cough in the background.
        """
        return self._max_score

    def last_energy(self) -> float:
        """
        Reports the Root Mean Square (RMS) energy of the latest audio frame.
        Unlike the AI-based VAD score, this is a purely mathematical measure
        of loudness. It acts as a reliable backup for speech detection,
        ensuring that even if the AI model is uncertain, loud sounds
        (which are often speech) will still be captured and processed
        by the transcription engine without being cut off early.
        """
        return self._last_energy

    def last_dynamic_threshold(self) -> float:
        """
        Returns the current sensitivity threshold for energy-based detection.
        This value is calculated based on the learned 'noise floor' of the
        user's environment. By returning this dynamic value, the system
        allows the caller to monitor how well the VAD is adapting to its
        surroundings, which is critical for maintaining performance in
        varying acoustic conditions like home offices versus busy cafes.
        """
        return self._last_dynamic_threshold

    def flush(self) -> bytes:
        """
        Retrieves the complete buffered utterance and clears the gate's state.
        This is the final step in the utterance lifecycle, where all the
        captured speech bytes are handed off to the Ear for processing.
        After calling flush, the gate is automatically reset, making it
        immediately ready to begin capturing and analyzing the next
        spoken sentence without any further manual intervention.
        """
        audio = bytes(self._buffer)
        self.reset()
        return audio
