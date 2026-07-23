import os
from unittest.mock import MagicMock, patch

import numpy
import pytest

try:
    import sherpa_onnx
except (ModuleNotFoundError, ImportError, OSError) as exc:
    import sys
    if sys.platform == "darwin":
        try:
            import ctypes
            import onnxruntime
            capi_dir = os.path.dirname(onnxruntime.capi.__file__)
            for f in os.listdir(capi_dir):
                if f.startswith("libonnxruntime") and f.endswith(".dylib"):
                    ctypes.CDLL(os.path.join(capi_dir, f), mode=ctypes.RTLD_GLOBAL)
                    break
            import sherpa_onnx
        except Exception:
            raise exc
    else:
        raise exc

import src.models.parakeet as parakeet


def test_transcribe():
    data = "hello_world".encode("utf-8")
    end = (len(data) // 8) * 8
    audio = numpy.frombuffer(data[:end])

    stream_mock = MagicMock()
    recognizer = MagicMock()

    stream_mock.result.text.strip.return_value = "hello_world"
    recognizer.create_stream.return_value = stream_mock

    text = parakeet.transcribe(recognizer, audio)
    assert text == "hello_world"


def test_download_and_extract():
    mock_tar = MagicMock()
    mock_tar.__enter__.return_value = mock_tar

    url = "https://example.com/model.tar.bz2"
    dest_dir = "/dummy/dest"

    with patch("os.makedirs") as mock_makedirs, \
         patch("urllib.request.urlretrieve") as mock_retrieve, \
         patch("os.remove") as mock_remove, \
         patch("os.path.exists", return_value=True), \
         patch("tarfile.open", return_value=mock_tar) as mock_tarfile_open:

        parakeet._download_and_extract(url, dest_dir)

        mock_makedirs.assert_called_once_with(dest_dir, exist_ok=True)
        mock_retrieve.assert_called_once_with(url, "/dummy/dest/model.tar.bz2")
        mock_tarfile_open.assert_called_once_with("/dummy/dest/model.tar.bz2", "r:bz2")
        mock_tar.extractall.assert_called_once_with(path=dest_dir)
        mock_remove.assert_called_once_with("/dummy/dest/model.tar.bz2")


def test_load_model_already_exists():
    with patch("os.path.exists", return_value=True), \
         patch("src.models.parakeet._download_and_extract") as mock_download, \
         patch.object(sherpa_onnx.OfflineRecognizer, "from_moonshine", return_value="<mock_recognizer>"):

        recognizer = parakeet.load_model("moonshine-base")
        assert recognizer == "<mock_recognizer>"
        mock_download.assert_not_called()


@pytest.mark.parametrize(
    "model_name, folder_name, factory_method, expected_kwargs",
    [
        (
            "moonshine-base",
            "sherpa-onnx-moonshine-base-en-int8",
            "from_moonshine",
            {
                "preprocessor": "{cache_path}/preprocess.onnx",
                "encoder": "{cache_path}/encode.int8.onnx",
                "uncached_decoder": "{cache_path}/uncached_decode.int8.onnx",
                "cached_decoder": "{cache_path}/cached_decode.int8.onnx",
                "tokens": "{cache_path}/tokens.txt",
                "num_threads": 6,
                "debug": False,
            },
        ),
        (
            "sensevoice",
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "from_sense_voice",
            {
                "model": "{cache_path}/model.int8.onnx",
                "tokens": "{cache_path}/tokens.txt",
                "use_itn": True,
                "language": "en",
            },
        ),
        (
            "nemo-ctc",
            "sherpa-onnx-nemo-nemo-ctc-int8",
            "from_nemo_ctc",
            {
                "model": "{cache_path}/model.int8.onnx",
                "tokens": "{cache_path}/tokens.txt",
                "num_threads": 6,
                "sample_rate": 16000,
                "feature_dim": 80,
                "debug": False,
            },
        ),
        (
            "nemo-parakeet-tdt-0.6b-v2",
            "sherpa-onnx-nemo-nemo-parakeet-tdt-0.6b-v2-int8",
            "from_transducer",
            {
                "encoder": "{cache_path}/encoder.int8.onnx",
                "decoder": "{cache_path}/decoder.int8.onnx",
                "joiner": "{cache_path}/joiner.int8.onnx",
                "tokens": "{cache_path}/tokens.txt",
                "num_threads": 6,
                "sample_rate": 16000,
                "feature_dim": 80,
                "decoding_method": "greedy_search",
                "model_type": "nemo_transducer",
                "debug": False,
            },
        ),
    ],
)
def test_load_model_types(monkeypatch, model_name, folder_name, factory_method, expected_kwargs):
    monkeypatch.setattr(parakeet.settings, "stt_model", model_name)
    monkeypatch.setattr(parakeet.settings, "rate", 16000)

    with patch("src.models.parakeet.get_integer_from_environment", return_value=6), \
         patch("src.models.parakeet._download_and_extract") as mock_download, \
         patch("os.path.exists", return_value=False), \
         patch.object(sherpa_onnx.OfflineRecognizer, factory_method, return_value="<mock_recognizer>") as mock_factory:

        recognizer = parakeet.load_model(model_name)

        assert recognizer == "<mock_recognizer>"
        mock_download.assert_called_once()

        home = os.path.expanduser("~")
        cache_path = f"{home}/.cache/parakeet-flow/models/{folder_name}"
        formatted_kwargs = {
            k: v.format(cache_path=cache_path) if isinstance(v, str) and "{cache_path}" in v else v
            for k, v in expected_kwargs.items()
        }
        mock_factory.assert_called_once_with(**formatted_kwargs)
