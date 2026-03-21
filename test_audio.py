import socket
import time
import os
import wave
import sys

SOCKET_PATH = "/tmp/parakeet.sock"

def test_transcription(wav_file_path):
    if not os.path.exists(SOCKET_PATH):
        print("Error: Socket not found. Make sure brain.py is running.")
        sys.exit(1)

    print(f"Reading {wav_file_path}...")
    with wave.open(wav_file_path, 'rb') as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
            print("Warning: Audio file is not 16kHz Mono 16-bit PCM. The model may output gibberish.")
        
        audio_data = wf.readframes(wf.getnframes())

    print(f"Sending {len(audio_data)} bytes to Brain for transcription...")
    
    start_time = time.time()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(SOCKET_PATH)
        client.sendall(audio_data)
        client.close()
        
        print("Data sent! The brain should type the transcription shortly.")
        print("Note: Switch your cursor focus to a text editor NOW to see the result.")
    except Exception as e:
        print(f"Failed to connect to Brain: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_audio.py <path_to_16khz_mono.wav>")
    else:
        test_transcription(sys.argv[1])