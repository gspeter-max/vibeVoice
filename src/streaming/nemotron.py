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
# This section loads a system library to make finding the highest number faster.
try:
    ACCELERATE_LIBRARY = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
    # vDSP_maxvi is a function that finds the largest number and its position in a list.
    ACCELERATE_LIBRARY.vDSP_maxvi.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_long,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong
    ]
    HAS_HARDWARE_ACCELERATION = True
except Exception:
    HAS_HARDWARE_ACCELERATION = False

def find_maximum_value_index(numbers_array):
    """
    Finds the position of the largest number in an array.
    
    Args:
        numbers_array (np.ndarray): The list of numbers to check.
        
    Returns:
        int: The index position of the largest number.
    """
    if not HAS_HARDWARE_ACCELERATION: 
        return np.argmax(numbers_array)
    
    # Ensure the data is in the correct format for the hardware library
    contiguous_numbers = np.ascontiguousarray(numbers_array.flatten(), dtype=np.float32)
    maximum_value = ctypes.c_float()
    maximum_index = ctypes.c_ulong()
    
    # Use the hardware library to find the maximum value and its index
    ACCELERATE_LIBRARY.vDSP_maxvi(
        contiguous_numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        1, 
        ctypes.byref(maximum_value), 
        ctypes.byref(maximum_index), 
        contiguous_numbers.size
    )
    return int(maximum_index.value)

class AudioSpectrogramConverter:
    """
    Converts raw sound data into a mathematical picture called a spectrogram.
    The Nemotron model needs this picture to understand the sound.
    """
    def __init__(self, model_configuration, filterbank_file_path):
        """
        Sets up the converter with settings and a pre-made frequency map.
        
        Args:
            model_configuration (dict): Settings for processing sound.
            filterbank_file_path (str): Path to the frequency map file.
        """
        self.settings = model_configuration['preprocessor']
        # Create a mathematical window to look at small pieces of sound
        self.mathematical_window = get_window(
            self.settings['window'], 
            self.settings['win_length'], 
            fftbins=True
        )
        
        # Load the frequency map (filterbank) from a binary file
        self.frequency_map = np.fromfile(filterbank_file_path, dtype=np.float32).reshape(1, 128, 257)[0]

    def convert_sound_to_spectrogram(self, raw_sound_data):
        """
        Changes raw sound samples into a log-mel spectrogram picture.
        
        Args:
            raw_sound_data (np.ndarray): The sound samples as numbers.
            
        Returns:
            np.ndarray: A 3D array representing the sound picture.
        """
        # Step 1: Make high sounds clearer
        processed_sound = np.append(
            raw_sound_data[0], 
            raw_sound_data[1:] - self.settings['preemph'] * raw_sound_data[:-1]
        )
        
        # Step 2: Cut the sound into small overlapping pieces
        number_of_frames = 1 + (len(processed_sound) - self.settings['win_length']) // self.settings['hop_length']
        sound_frames = np.lib.stride_tricks.as_strided(
            processed_sound, 
            shape=(number_of_frames, self.settings['win_length']), 
            strides=(processed_sound.strides[0] * self.settings['hop_length'], processed_sound.strides[0])
        )
        
        # Step 3: Use a mathematical formula (FFT) to find the volume of different pitches
        pitch_volumes = np.abs(np.fft.rfft(sound_frames * self.mathematical_window, n=self.settings['n_fft'])) ** 2
        
        # Step 4: Map the pitches to the frequency map and take the logarithm
        mel_volumes = np.dot(pitch_volumes, self.frequency_map.T)
        log_mel_spectrogram = np.log(mel_volumes + 1e-5)
        
        # Return the final picture in the shape the model expects
        return log_mel_spectrogram.T[np.newaxis, :, :].astype(np.float32)

def download_nemotron_model(cache_directory_path):
    """
    Downloads the Nemotron AI model from Hugging Face.
    It only downloads the specific files needed to run the AI.
    """
    from huggingface_hub import snapshot_download
    
    log.info("⬇️ Downloading Nemotron-0.6B model (this may take a minute)...")
    snapshot_download(
        repo_id="danielbodart/nemotron-speech-600m-onnx",
        local_dir=cache_directory_path,
        allow_patterns=[
            "int8-dynamic/*", 
            "shared/*", 
            "config.json"
        ]
    )
    log.info("✅ Download complete.")


class NemotronStreamingEngine:
    """
    The main engine that turns spoken sound into written text in real-time.
    It remembers what was said before so the text stays smooth.
    """
    def __init__(self):
        """
        Loads the AI models and the list of words (tokens).
        If the model isn't on the computer yet, it downloads it automatically.
        """
        # Find where the model files are stored on the computer
        local_models_directory = os.path.join(os.getcwd(), "models/nemotron-0.6b-onnx")
        cache_models_directory = os.path.expanduser("~/.cache/parakeet-flow/models/nemotron-0.6b-onnx")
        
        if os.path.exists(local_models_directory):
            self.models_directory = local_models_directory
        else:
            self.models_directory = cache_models_directory
            # If the config file isn't in the cache, it means the model isn't downloaded yet.
            config_file_path = os.path.join(self.models_directory, "config.json")
            if not os.path.exists(config_file_path):
                log.info(f"Model not found. Initiating auto-download to {self.models_directory}...")
                download_nemotron_model(self.models_directory)
                
        log.info(f"[Nemotron] Loading from: {self.models_directory}")
        
        # Load the configuration file which has settings for the AI
        with open(os.path.join(self.models_directory, "config.json"), "r") as config_file:
            self.configuration = json.load(config_file)
            
        # Load the two parts of the AI: the Encoder and the Decoder
        self.encoder_model = ort.InferenceSession(
            os.path.join(self.models_directory, "int8-dynamic/encoder_model.onnx"), 
            providers=["CPUExecutionProvider"]
        )
        self.decoder_model = ort.InferenceSession(
            os.path.join(self.models_directory, "int8-dynamic/decoder_model.onnx"), 
            providers=["CPUExecutionProvider"]
        )
        
        # Load the list of text pieces (tokens) that the AI can write
        tokens_file_path = os.path.join(self.models_directory, "shared/tokens.txt")
        with open(tokens_file_path, "r", encoding="utf-8") as tokens_file:
            self.vocabulary_tokens = [line.strip().split()[0] for line in tokens_file if line.strip()]
            
        # Set up the sound-to-picture converter
        filterbank_file_path = os.path.join(self.models_directory, "shared/filterbank.bin")
        self.sound_processor = AudioSpectrogramConverter(self.configuration, filterbank_file_path)
        
        # Clear all memory to start fresh
        self.clear_internal_memory()

    def clear_internal_memory(self):
        """
        Clears the engine's memory so it is ready for a new person to speak.
        """
        encoder_settings = self.configuration['encoder']
        decoder_settings = self.configuration['decoder']
        
        # Memory for the Encoder part of the AI
        self.encoder_memory = {
            "cache_last_channel": np.zeros(encoder_settings['cache_last_channel_shape'], dtype=np.float32),
            "cache_last_time": np.zeros(encoder_settings['cache_last_time_shape'], dtype=np.float32),
            "cache_last_channel_len": np.array([0], dtype=np.int64)
        }
        
        # Memory for the Decoder part of the AI
        self.decoder_memory = {
            "input_states_1": np.zeros((2, 1, decoder_settings['prediction_hidden']), dtype=np.float32),
            "input_states_2": np.zeros((2, 1, decoder_settings['prediction_hidden']), dtype=np.float32)
        }
        
        # The last piece of text the AI wrote (starts at 0)
        self.last_written_token = np.array([[0]], dtype=np.int32)
        self.full_text_result = ""
        
        # Small piece of sound saved from the last time to make the transition smooth
        self.previous_sound_frames = np.zeros((1, 128, 9), dtype=np.float32)

    def add_audio_chunk_and_get_text(self, new_sound_data):
        """
        Takes a new piece of sound and adds the new words to the text result.
        
        Args:
            new_sound_data (np.ndarray): The new piece of sound.
            
        Returns:
            str: All the text the AI has written so far in this session.
        """
        # Convert the new sound into a mathematical picture
        current_spectrogram = self.sound_processor.convert_sound_to_spectrogram(new_sound_data)
        
        # Combine the new picture with the small piece saved from last time
        combined_spectrogram = np.concatenate([self.previous_sound_frames, current_spectrogram], axis=2)
        self.previous_sound_frames = current_spectrogram[:, :, -9:] # Save the end of this picture for next time
        
        # Step 1: Run the Encoder part of the AI
        encoder_inputs = {
            "audio_signal": combined_spectrogram, 
            "length": np.array([combined_spectrogram.shape[2]], dtype=np.int64), 
            **self.encoder_memory
        }
        encoder_outputs = self.encoder_model.run(None, encoder_inputs)
        
        # Update the Encoder's memory with what it just learned
        self.encoder_memory.update({
            "cache_last_channel": encoder_outputs[2], 
            "cache_last_time": encoder_outputs[3], 
            "cache_last_channel_len": encoder_outputs[4]
        })
        
        # Step 2: Check each frame of the picture to find new words
        encoded_frames = encoder_outputs[0]
        blank_token_id = self.configuration['decoder']['blank_id']
        max_words_per_frame = self.configuration['decoder']['max_symbols_per_frame']
        
        for time_step in range(encoded_frames.shape[2]):
            single_frame = encoded_frames[:, :, time_step:time_step+1]
            
            # The AI can find multiple words in one small piece of sound
            for _ in range(max_words_per_frame):
                decoder_inputs = {
                    "encoder_outputs": single_frame, 
                    "targets": self.last_written_token, 
                    "target_length": np.array([1], dtype=np.int32), 
                    **self.decoder_memory
                }
                decoder_outputs = self.decoder_model.run(None, decoder_inputs)
                
                # Find the most likely word piece (token)
                predicted_token_id = find_maximum_value_index(decoder_outputs[0][0, 0, 0])
                
                if predicted_token_id == blank_token_id:
                    # If the AI says 'blank', it means there are no more words in this frame
                    break
                
                # Turn the token ID into a piece of written text
                token_text = self.vocabulary_tokens[predicted_token_id]
                
                # If the token starts with a special space character, add a space
                if token_text.startswith("▁"):
                    self.full_text_result += " " + token_text[1:]
                else:
                    self.full_text_result += token_text
                
                # Update the Decoder's memory so it knows what it just wrote
                self.last_written_token = np.array([[predicted_token_id]], dtype=np.int32)
                self.decoder_memory.update({
                    "input_states_1": decoder_outputs[2], 
                    "input_states_2": decoder_outputs[3]
                })
                
        return self.full_text_result
