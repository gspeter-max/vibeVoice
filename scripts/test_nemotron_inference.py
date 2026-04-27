import onnxruntime as ort
import numpy as np
import wave
import json
import os
import ctypes
from scipy.signal import get_window

# Load Apple's Accelerate framework for hardware-accelerated math
try:
    ACCELERATE = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
    # void vDSP_maxvi(const float *__v1, vDSP_Stride __v2, float *__v3, vDSP_Length *__v4, vDSP_Length __v5);
    ACCELERATE.vDSP_maxvi.argtypes = [
        ctypes.POINTER(ctypes.c_float), # Input vector
        ctypes.c_long,                 # Stride
        ctypes.POINTER(ctypes.c_float), # Output max value
        ctypes.POINTER(ctypes.c_ulong), # Output max index
        ctypes.c_ulong                  # Length
    ]
    HAS_ACCELERATE = True
except Exception:
    HAS_ACCELERATE = False

def optimized_argmax(arr):
    if not HAS_ACCELERATE:
        return np.argmax(arr)
    
    # vDSP needs contiguous float32 data
    arr = np.ascontiguousarray(arr.flatten(), dtype=np.float32)
    max_val = ctypes.c_float()
    max_idx = ctypes.c_ulong()
    
    ACCELERATE.vDSP_maxvi(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        1,
        ctypes.byref(max_val),
        ctypes.byref(max_idx),
        arr.size
    )
    return int(max_idx.value)

class MelPreprocessor:
    def __init__(self, config, filterbank_path):
        self.cfg = config['preprocessor']
        self.sample_rate = config['audio']['sample_rate']
        self.n_mels = self.cfg['n_mels']
        self.n_fft = self.cfg['n_fft']
        self.hop_length = self.cfg['hop_length']
        self.win_length = self.cfg['win_length']
        self.window = get_window(self.cfg['window'], self.win_length, fftbins=True)
        
        # Load filterbank
        # Shape is [1, 128, 257]
        self.fb = np.fromfile(filterbank_path, dtype=np.float32).reshape(1, 128, 257)
        self.fb = self.fb[0] # [128, 257]

    def process(self, audio):
        # Pre-emphasis
        audio = np.append(audio[0], audio[1:] - self.cfg['preemph'] * audio[:-1])
        
        # Framing
        n_frames = 1 + (len(audio) - self.win_length) // self.hop_length
        frames = np.lib.stride_tricks.as_strided(
            audio, 
            shape=(n_frames, self.win_length), 
            strides=(audio.strides[0] * self.hop_length, audio.strides[0])
        )
        
        # Windowing
        frames = frames * self.window
        
        # RFFT
        stft = np.fft.rfft(frames, n=self.n_fft)
        mag = np.abs(stft) ** 2
        
        # Mel Filterbank
        mel_spec = np.dot(mag, self.fb.T) # [n_frames, 128]
        
        # Log Mel
        log_mel = np.log(mel_spec + 1e-5)
        
        # Band-major layout [1, 128, n_frames]
        return log_mel.T[np.newaxis, :, :].astype(np.float32)

class NemotronInference:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            self.cfg = json.load(f)
        
        self.encoder = ort.InferenceSession(
            os.path.join(model_dir, "int8-dynamic/encoder_model.onnx"),
            providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            os.path.join(model_dir, "int8-dynamic/decoder_model.onnx"),
            providers=["CPUExecutionProvider"]
        )
        
        # Load tokens
        self.tokens = []
        with open(os.path.join(model_dir, "shared/tokens.txt"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                # The token is the first part, unless it's a space character
                self.tokens.append(parts[0])
        
        self.preprocessor = MelPreprocessor(self.cfg, os.path.join(model_dir, "shared/filterbank.bin"))
        
        # Initial states
        self.reset_states()

    def reset_states(self):
        e_cfg = self.cfg['encoder']
        self.enc_cache = {
            "cache_last_channel": np.zeros(e_cfg['cache_last_channel_shape'], dtype=np.float32),
            "cache_last_time": np.zeros(e_cfg['cache_last_time_shape'], dtype=np.float32),
            "cache_last_channel_len": np.array([0], dtype=np.int64)
        }
        
        d_cfg = self.cfg['decoder']
        self.dec_states = {
            "input_states_1": np.zeros((2, 1, d_cfg['prediction_hidden']), dtype=np.float32),
            "input_states_2": np.zeros((2, 1, d_cfg['prediction_hidden']), dtype=np.float32)
        }
        self.last_token = np.array([[0]], dtype=np.int32) # Start with 0
        self.transcript = ""
        if hasattr(self, 'mel_cache'):
            del self.mel_cache

    def transcribe(self, wav_path, chunk_duration_ms=560):
        self.reset_states()
        with wave.open(wav_path, "rb") as f:
            samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            audio = samples.astype(np.float32) / 32768.0
        
        # Calculate chunk samples based on duration
        # 16000 samples per second -> 16 samples per ms
        chunk_samples = int(chunk_duration_ms * 16)
        num_chunks = len(audio) // chunk_samples
        
        print(f"  [Config] Chunk size: {chunk_duration_ms}ms ({chunk_samples} samples)")
        
        for i in range(num_chunks):
            chunk = audio[i*chunk_samples : (i+1)*chunk_samples]
            self.process_chunk(chunk)
            
        return self.transcript

    def process_chunk(self, audio_chunk):
        # Preprocess
        mel = self.preprocessor.process(audio_chunk) # [1, 128, n_frames]
        n_frames = mel.shape[2]
        
        # Encoder
        if not hasattr(self, 'mel_cache'):
            self.mel_cache = np.zeros((1, 128, 9), dtype=np.float32)
        
        full_mel = np.concatenate([self.mel_cache, mel], axis=2)
        self.mel_cache = mel[:, :, -9:] # Update cache
        
        total_frames = full_mel.shape[2]
        
        enc_inputs = {
            "audio_signal": full_mel,
            "length": np.array([total_frames], dtype=np.int64),
            **self.enc_cache
        }
        
        enc_outputs = self.encoder.run(None, enc_inputs)
        
        self.enc_cache["cache_last_channel"] = enc_outputs[2]
        self.enc_cache["cache_last_time"] = enc_outputs[3]
        self.enc_cache["cache_last_channel_len"] = enc_outputs[4]
        
        encoded_frames = enc_outputs[0] # [1, 1024, T]
        T = encoded_frames.shape[2]
        
        # Decoder loop
        blank_id = self.cfg['decoder']['blank_id']
        
        for t in range(T):
            frame = encoded_frames[:, :, t:t+1]
            
            for _ in range(self.cfg['decoder']['max_symbols_per_frame']):
                dec_inputs = {
                    "encoder_outputs": frame,
                    "targets": self.last_token,
                    "target_length": np.array([1], dtype=np.int32),
                    **self.dec_states
                }
                
                dec_outputs = self.decoder.run(None, dec_inputs)
                logits = dec_outputs[0]
                token = optimized_argmax(logits[0, 0, 0])
                
                if token == blank_id:
                    break
                
                # Emit token
                token_str = self.tokens[token]
                # Special space character handling
                if token_str.startswith("▁"):
                    self.transcript += " " + token_str[1:]
                else:
                    self.transcript += token_str
                
                self.last_token = np.array([[token]], dtype=np.int32)
                self.dec_states["input_states_1"] = dec_outputs[2]
                self.dec_states["input_states_2"] = dec_outputs[3]

if __name__ == "__main__":
    wav_path = "/Users/apple/project/vibeVoice/logs/streaming_sessions/2026-04-24_13-34-59_663d1685a0b94a5393f9c4dbe09a788c_audio/rec1_chunk0.wav"
    model_dir = "models/nemotron-0.6b-onnx"
    
    print(f"Loading model from {model_dir}...")
    infer = NemotronInference(model_dir)
    
    for duration in [560, 1120]:
        print(f"\n--- Testing with {duration}ms chunks ---")
        import time
        start = time.time()
        result = infer.transcribe(wav_path, chunk_duration_ms=duration)
        end = time.time()
        
        print(f"  [Result] {result.strip()}")
        print(f"  [Time] {end - start:.2f}s")
    print("\n")
