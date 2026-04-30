"""
colab_manifest_builder.py

This module contains utilities to build NeMo format JSON manifests 
for the Parakeet model fine-tuning pipeline. 
It parses audio files and transcripts to generate the required `.json` format.
"""

import os
import json
import wave
from pathlib import Path

def get_wav_duration(wav_file_path: str) -> float:
    """
    Reads a .wav file and calculates its duration in seconds.
    
    Args:
        wav_file_path: The absolute or relative path to the .wav file.
        
    Returns:
        The duration of the audio in seconds.
        
    Raises:
        FileNotFoundError: If the .wav file does not exist.
        wave.Error: If the file is not a valid .wav format.
    """
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"Audio file not found: {wav_file_path}")
        
    with wave.open(wav_file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def build_nemo_manifest_from_directory(
    audio_directory_path: str, 
    transcript_mapping: dict, 
    output_manifest_file_path: str
) -> None:
    """
    Reads all audio files in the specified directory, matches them with 
    their corresponding transcript, calculates the duration, and writes 
    the result to a NeMo-compatible JSON manifest file.
    
    Args:
        audio_directory_path: The folder containing the .wav files.
        transcript_mapping: A dictionary where the key is the audio file name 
                            (without extension) and the value is the text transcript.
        output_manifest_file_path: The path where the output .json file will be saved.
    """
    audio_dir = Path(audio_directory_path)
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {audio_directory_path}")

    manifest_entries = []
    
    # Iterate through all wav files in the directory
    for audio_file in audio_dir.glob("*.wav"):
        file_name_without_extension = audio_file.stem
        
        # Check if we have a transcript for this audio file
        if file_name_without_extension not in transcript_mapping:
            continue
            
        text = transcript_mapping[file_name_without_extension]
        duration = get_wav_duration(str(audio_file))
        
        # Build the dictionary for this single audio file
        entry = {
            "audio_filepath": str(audio_file.absolute()),
            "duration": duration,
            "text": text
        }
        manifest_entries.append(entry)
        
    # Write all entries to the manifest file as JSON Lines
    with open(output_manifest_file_path, 'w', encoding='utf-8') as manifest_file:
        for entry in manifest_entries:
            json_line = json.dumps(entry)
            manifest_file.write(json_line + '\n')
