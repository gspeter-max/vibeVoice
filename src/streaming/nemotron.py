# src/streaming/nemotron.py
import os
import json
import wave
import ctypes
import numpy as np
import onnxruntime as ort
from scipy.signal import get_window
from src import log

# Hardware Acceleration: Apple Accelerate
# This section attempts to load the Accelerate framework for faster argmax operations on macOS.
try:
    ACCELERATE = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
    # void vDSP_maxvi(const float *__v1, vDSP_Stride __v2, float *__v3, vDSP_Length *__v4, vDSP_Length __v5);
    ACCELERATE.vDSP_maxvi.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_long,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong
    ]
    HAS_ACCELERATE = True
except Exception:
    HAS_ACCELERATE = False

def optimized_argmax(arr):
    """
    Finds the index of the maximum value in a float array using hardware acceleration if available.
    
    Args:
        arr (np.ndarray): Input array to find the argmax of.
        
    Returns:
        int: The index of the maximum value.
    """
    if not HAS_ACCELERATE: 
        return np.argmax(arr)
    
    # vDSP needs contiguous float32 data
    arr = np.ascontiguousarray(arr.flatten(), dtype=np.float32)
    max_val, max_idx = ctypes.c_float(), ctypes.c_ulong()
    
    # Use vDSP_maxvi for hardware-accelerated maximum value and index search
    ACCELERATE.vDSP_maxvi(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        1, 
        ctypes.byref(max_val), 
        ctypes.byref(max_idx), 
        arr.size
    )
    return int(max_idx.value)

class MelPreprocessor:
    """
    Preprocesses raw audio waveforms into Log-Mel Spectrograms.
    Follows the specific configuration required by the Nemotron model.
    """
    def __init__(self, config, filterbank_path):
        """
        Initializes the preprocessor with model config and pre-computed filterbanks.
        
        Args:
            config (dict): The model configuration containing preprocessor settings.
            filterbank_path (str): Path to the binary filterbank file.
        """
        self.cfg = config['preprocessor']
        # Create a Hann window for the STFT calculation
        self.window = get_window(self.cfg['window'], self.cfg['win_length'], fftbins=True)
        
        # Load the pre-computed Mel filterbank (expected shape [1, 128, 257])
        self.fb = np.fromfile(filterbank_path, dtype=np.float32).reshape(1, 128, 257)[0]

    def process(self, audio):
        """
        Converts raw audio to a Log-Mel Spectrogram.
        
        Args:
            audio (np.ndarray): Raw float32 audio samples.
            
        Returns:
            np.ndarray: Log-Mel Spectrogram with shape [1, n_mels, n_frames].
        """
        # Pre-emphasis filter to boost high frequencies
        audio = np.append(audio[0], audio[1:] - self.cfg['preemph'] * audio[:-1])
        
        # Framing: segment audio into overlapping frames
        n_frames = 1 + (len(audio) - self.cfg['win_length']) // self.cfg['hop_length']
        frames = np.lib.stride_tricks.as_strided(
            audio, 
            shape=(n_frames, self.cfg['win_length']), 
            strides=(audio.strides[0] * self.cfg['hop_length'], audio.strides[0])
        )
        
        # STFT: Windowing and Real FFT
        mag = np.abs(np.fft.rfft(frames * self.window, n=self.cfg['n_fft'])) ** 2
        
        # Apply Mel Filterbank and take the Log
        mel_spec = np.dot(mag, self.fb.T)
        log_mel = np.log(mel_spec + 1e-5)
        
        # Reshape to [1, 128, n_frames] as expected by the encoder
        return log_mel.T[np.newaxis, :, :].astype(np.float32)

class NemotronStreamingBackend:
    """
    NVIDIA Nemotron 0.6B RNN-T Streaming Backend.
    Maintains internal states for the encoder and decoder to allow continuous transcription.
    """
    def __init__(self):
        """
        Initializes the ONNX sessions and loads the model configuration and tokens.
        """
        # Prioritize local models directory if it exists
        local_dir = os.path.join(os.getcwd(), "models/nemotron-0.6b-onnx")
        if os.path.exists(local_dir):
            self.model_dir = local_dir
        else:
            self.model_dir = os.path.expanduser("~/.cache/parakeet-flow/models/nemotron-0.6b-onnx")
            
        log.info(f"[Nemotron] Initializing from: {self.model_dir}")
        
        # Load Model Configuration
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "r") as f:
            self.cfg = json.load(f)
            
        # Initialize ONNX Runtime sessions for Encoder and Decoder
        self.encoder = ort.InferenceSession(
            os.path.join(self.model_dir, "int8-dynamic/encoder_model.onnx"), 
            providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            os.path.join(self.model_dir, "int8-dynamic/decoder_model.onnx"), 
            providers=["CPUExecutionProvider"]
        )
        
        # Load vocabulary tokens
        tokens_path = os.path.join(self.model_dir, "shared/tokens.txt")
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens = [line.strip().split()[0] for line in f if line.strip()]
            
        # Initialize Preprocessor
        filterbank_path = os.path.join(self.model_dir, "shared/filterbank.bin")
        self.preprocessor = MelPreprocessor(self.cfg, filterbank_path)
        
        # Initialize streaming states
        self.reset()

    def reset(self):
        """
        Resets the internal states of the encoder, decoder, and transcript.
        Must be called between independent transcription sessions.
        """
        e_cfg, d_cfg = self.cfg['encoder'], self.cfg['decoder']
        
        # Encoder states (Caches for the Conformer/RNN layers)
        self.enc_cache = {
            "cache_last_channel": np.zeros(e_cfg['cache_last_channel_shape'], dtype=np.float32),
            "cache_last_time": np.zeros(e_cfg['cache_last_time_shape'], dtype=np.float32),
            "cache_last_channel_len": np.array([0], dtype=np.int64)
        }
        
        # Decoder states (LSTM hidden and cell states)
        self.dec_states = {
            "input_states_1": np.zeros((2, 1, d_cfg['prediction_hidden']), dtype=np.float32),
            "input_states_2": np.zeros((2, 1, d_cfg['prediction_hidden']), dtype=np.float32)
        }
        
        # RNN-T decoding state
        self.last_token = np.array([[0]], dtype=np.int32) # Starting token (usually 0/blank)
        self.transcript = ""
        
        # Mel Cache: necessary because of the overlapping nature of the preprocessor
        self.mel_cache = np.zeros((1, 128, 9), dtype=np.float32)

    def transcribe_chunk(self, audio):
        """
        Processes a single chunk of audio and appends discovered text to the transcript.
        
        Args:
            audio (np.ndarray): Raw float32 audio chunk.
            
        Returns:
            str: The full cumulative transcript for the current session.
        """
        # Convert audio to Mel Spectrogram
        mel = self.preprocessor.process(audio)
        
        # Prepend cached mel frames from the previous chunk to ensure temporal continuity
        full_mel = np.concatenate([self.mel_cache, mel], axis=2)
        self.mel_cache = mel[:, :, -9:] # Update cache with the last 9 frames
        
        # Run Encoder
        enc_inputs = {
            "audio_signal": full_mel, 
            "length": np.array([full_mel.shape[2]], dtype=np.int64), 
            **self.enc_cache
        }
        enc_outputs = self.encoder.run(None, enc_inputs)
        
        # Update Encoder Cache
        self.enc_cache.update({
            "cache_last_channel": enc_outputs[2], 
            "cache_last_time": enc_outputs[3], 
            "cache_last_channel_len": enc_outputs[4]
        })
        
        # Greedily decode each frame emitted by the encoder
        encoded_frames = enc_outputs[0]
        blank_id = self.cfg['decoder']['blank_id']
        
        for t in range(encoded_frames.shape[2]):
            frame = encoded_frames[:, :, t:t+1]
            
            # Sub-loop for RNN-T: allow multiple symbols to be emitted per frame
            for _ in range(self.cfg['decoder']['max_symbols_per_frame']):
                dec_inputs = {
                    "encoder_outputs": frame, 
                    "targets": self.last_token, 
                    "target_length": np.array([1], dtype=np.int32), 
                    **self.dec_states
                }
                dec_outputs = self.decoder.run(None, dec_inputs)
                
                # Get the most likely token
                token = optimized_argmax(dec_outputs[0][0, 0, 0])
                
                if token == blank_id:
                    # Break the sub-loop if a blank symbol is predicted
                    break
                
                # Process the predicted token
                t_str = self.tokens[token]
                # '▁' (u2581) is the SentencePiece separator indicating a new word/space
                self.transcript += (" " + t_str[1:] if t_str.startswith("▁") else t_str)
                
                # Update Decoder States
                self.last_token = np.array([[token]], dtype=np.int32)
                self.dec_states.update({
                    "input_states_1": dec_outputs[2], 
                    "input_states_2": dec_outputs[3]
                })
                
        return self.transcript
