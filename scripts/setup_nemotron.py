from huggingface_hub import snapshot_download
import os

def download_model():
    model_id = "danielbodart/nemotron-speech-600m-onnx"
    local_dir = "models/nemotron-0.6b-onnx"
    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        allow_patterns=["int8-dynamic/*", "tokens.txt", "config.json"]
    )
    print("Download complete.")

if __name__ == "__main__":
    download_model()
