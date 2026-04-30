import os
import json
import wave
import pytest
from pathlib import Path
from src.data_record.colab_manifest_builder import (
    build_nemo_manifest_from_directory,
    get_wav_duration
)

def create_mock_wav(path: Path, frames: int, framerate: int = 16000):
    """Helper to create real valid wav files for testing."""
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(b'\x00\x00' * frames)

def test_build_nemo_manifest_from_directory_creates_valid_json_lines(tmp_path):
    """
    Test that the build_nemo_manifest_from_directory function correctly parses 
    a directory of audio files and a transcript dictionary to create a valid 
    NeMo format JSON manifest. 
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    
    wav_path_1 = audio_dir / "test_1.wav"
    create_mock_wav(wav_path_1, frames=16000) # 1 sec
        
    wav_path_2 = audio_dir / "test_2.wav"
    create_mock_wav(wav_path_2, frames=32000) # 2 secs

    transcripts = {
        "test_1": "hello world",
        "test_2": "testing nemo manifest"
    }
    
    output_manifest_path = tmp_path / "output_manifest.json"
    
    build_nemo_manifest_from_directory(
        audio_directory_path=str(audio_dir), 
        transcript_mapping=transcripts, 
        output_manifest_file_path=str(output_manifest_path)
    )
    
    assert output_manifest_path.exists()
    
    with open(output_manifest_path, 'r', encoding='utf-8') as manifest_file:
        lines = manifest_file.readlines()
        
    assert len(lines) == 2
    
    entries = sorted([json.loads(line) for line in lines], key=lambda x: x["audio_filepath"])
    
    entry_1 = entries[0]
    entry_2 = entries[1]
    
    assert entry_1["audio_filepath"] == str(wav_path_1.absolute())
    assert entry_1["duration"] == 1.0
    assert entry_1["text"] == "hello world"
    
    assert entry_2["audio_filepath"] == str(wav_path_2.absolute())
    assert entry_2["duration"] == 2.0
    assert entry_2["text"] == "testing nemo manifest"

def test_build_nemo_manifest_skips_missing_transcripts(tmp_path):
    """
    Test that audio files without a matching transcript are skipped 
    and not included in the output manifest.
    """
    audio_dir = tmp_path / "audio_missing"
    audio_dir.mkdir()
    
    wav_path_1 = audio_dir / "test_1.wav"
    create_mock_wav(wav_path_1, frames=16000)
    
    wav_path_unmapped = audio_dir / "unmapped.wav"
    create_mock_wav(wav_path_unmapped, frames=16000)
    
    transcripts = {
        "test_1": "hello world",
    }
    
    output_manifest_path = tmp_path / "output_missing.json"
    
    build_nemo_manifest_from_directory(
        audio_directory_path=str(audio_dir), 
        transcript_mapping=transcripts, 
        output_manifest_file_path=str(output_manifest_path)
    )
    
    with open(output_manifest_path, 'r', encoding='utf-8') as manifest_file:
        lines = manifest_file.readlines()
        
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["audio_filepath"] == str(wav_path_1.absolute())

def test_build_nemo_manifest_invalid_directory(tmp_path):
    """
    Test that providing an invalid directory raises a NotADirectoryError.
    """
    invalid_dir = tmp_path / "non_existent_dir"
    output_manifest_path = tmp_path / "output.json"
    
    with pytest.raises(NotADirectoryError):
        build_nemo_manifest_from_directory(
            audio_directory_path=str(invalid_dir), 
            transcript_mapping={}, 
            output_manifest_file_path=str(output_manifest_path)
        )

def test_get_wav_duration_file_not_found(tmp_path):
    """
    Test that get_wav_duration raises FileNotFoundError for missing files.
    """
    missing_wav = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        get_wav_duration(str(missing_wav))

