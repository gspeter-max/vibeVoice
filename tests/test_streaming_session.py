# tests/test_streaming_session.py
import pytest
import time
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

def test_session_deduplicates_incoming_text():
    session = StreamingSession()
    
    # Simulate Brain receiving chunks with more context
    clean_1, stats_1 = session.process_incoming_text_chunk(0, "Hello world this is")
    assert clean_1 == "Hello world this is"
    
    # Chunk 2 overlaps with "world this is"
    clean_2, stats_2 = session.process_incoming_text_chunk(1, "world this is great and working")
    
    # The session should remember "Hello world this is" and trim "world this is" from chunk 2
    # Result should be "great and working" (3 words, so safety check won't skip it)
    assert clean_2 == "great and working"
    assert session.get_full_transcript() == "Hello world this is great and working"
