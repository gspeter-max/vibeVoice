# tests/test_streaming_session.py
from src.streaming.session import StreamingSession

def test_session_tracks_tail_bytes_for_audio_overlap():
    session = StreamingSession(overlap_seconds=0.1, sample_rate=16000)
    
    # Fake audio: 1 second of zeros
    fake_audio = b"\x00\x00" * 16000 
    
    # First chunk shouldn't have any overlap added at the start
    processed_audio_1 = session.process_outgoing_audio_chunk(fake_audio, stop_session=False, silence_seconds=0.0)
    assert len(processed_audio_1) == len(fake_audio)
    
    # Second chunk should have the tail of the first chunk added to the start
    processed_audio_2 = session.process_outgoing_audio_chunk(fake_audio, stop_session=False, silence_seconds=0.0)
    
    # 0.1 seconds * 16000 samples * 2 bytes/sample = 3200 bytes overlap
    expected_overlap_bytes = int(0.1 * 16000 * 2)
    assert len(processed_audio_2) == len(fake_audio) + expected_overlap_bytes


