"""
Test Kokoro TTS Client
======================
Test Unix socket communication with Kokoro TTS service.
"""
import sys
import time
sys.path.insert(0, '.')

from core.socket_server import SocketClient
from core.audio_utils import get_wav_info


def test_kokoro_service():
    """Test Kokoro TTS service via Unix socket."""
    SOCKET_PATH = "/tmp/tts-kokoro.sock"

    print("🧪 Testing Kokoro TTS Service")
    print("=" * 40)
    print("")

    # Test texts
    test_cases = [
        "Hello, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Kokoro is a lightweight text-to-speech model.",
    ]

    client = SocketClient(SOCKET_PATH, timeout=10.0)

    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {text}")
        print(f"Text length: {len(text)} characters")

        try:
            # Send request
            start_time = time.time()
            audio_bytes = client.send_request(text)
            elapsed = time.time() - start_time

            # Get audio info
            info = get_wav_info(audio_bytes)

            print(f"✅ Success!")
            print(f"   Audio size: {info['bytes']} bytes")
            print(f"   Duration: {info['duration']:.2f} seconds")
            print(f"   Sample rate: {info['sample_rate']} Hz")
            print(f"   Response time: {elapsed:.2f} seconds")
            print(f"   Real-time factor: {elapsed / info['duration']:.2f}x")
            print("")

            # Save to file
            output_file = f"test_output_{i}.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)
            print(f"   Saved: {output_file}")
            print("")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("")

    print("=" * 40)
    print("✅ All tests completed!")
    print("")
    print("Play outputs with:")
    for i in range(1, len(test_cases) + 1):
        print(f"  afplay test_output_{i}.wav")


if __name__ == "__main__":
    try:
        test_kokoro_service()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
