"""
colab_training_pipeline.py

This script is a "Manager" or "Helper" to help you fine-tune the 
nvidia/parakeet-tdt-1.1b ASR model on Google Colab.

It prepares your computer (Colab) for training in 5 steps:
1. Downloads NVIDIA's official "speech_to_text_finetune.py" script.
2. Downloads NVIDIA's official training settings (YAML file).
3. Downloads the audio data (Clean Speech, Noise, and Echoes).
4. Creates the "Manifest" (The map that tells NeMo where the audio is).
5. Prepares the rules for LoRA (Low-Rank Adaptation) and Noise.

Target Environment: Google Colab T4 GPU
"""

# --- FASTEST INSTALLATION (Run this in a Colab cell first - takes ~30 seconds) ---
# !pip install -q uv
# !apt-get install aria2 -y
# !uv pip install --system "nemo_toolkit[asr]" peft bitsandbytes
# !apt-get install -y libsndfile1 ffmpeg sox
# ---------------------------------------------------------------------------------

import os
import subprocess
import argparse
import json
import yaml
import wave
from pathlib import Path

# --- File Paths for Colab ---
OFFICIAL_NEMO_SCRIPT_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/speech_to_text_finetune.py"
OFFICIAL_SCRIPT_NAME = "speech_to_text_finetune.py"

# We download an official config so we have the right optimizer and scheduler settings
OFFICIAL_CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml"
OFFICIAL_CONFIG_NAME = "base_config.yaml"

# Where the final improved model will be saved
DRIVE_SAVE_PATH = "/content/drive/MyDrive/parakeet_lora_tuned.nemo"


def download_official_nvidia_files(dry_run: bool = False):
    """
    Downloads the official training script and settings from NVIDIA's GitHub.
    
    If the files already exist, we skip the download to save time.
    """
    print(f"Checking for official NVIDIA files...")
    
    files_to_download = {
        OFFICIAL_SCRIPT_NAME: OFFICIAL_NEMO_SCRIPT_URL,
        OFFICIAL_CONFIG_NAME: OFFICIAL_CONFIG_URL
    }
    
    for filename, url in files_to_download.items():
        if os.path.exists(filename):
            print(f"  -> {filename} already exists. Skipping download.")
            continue
            
        print(f"  -> Grabbing {filename}...")
        if not dry_run:
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-o", filename, url], check=True)
    
    if not dry_run:
        print("Official files ready.")


def download_and_extract_speech_data(dry_run: bool = False):
    """
    Downloads and extracts clean speech, background noise, and room echoes.
    
    If a dataset folder already exists, we skip it.
    """
    print("Checking for audio datasets...")
    
    # Map dataset names to their target extraction folder or file
    # (Checking for a folder is better than checking for the archive)
    datasets = {
        "LibriSpeech": ("librispeech_clean", "http://www.openslr.org/resources/12/dev-clean.tar.gz"),
        "musan": ("musan_noise", "http://www.openslr.org/resources/17/musan.tar.gz"),
        "RIRS_NOISES": ("room_echoes", "http://www.openslr.org/resources/28/rirs_noises.zip")
    }
    
    for folder_name, (key, url) in datasets.items():
        if os.path.exists(folder_name):
            print(f"  -> Dataset folder '{folder_name}' already exists. Skipping.")
            continue
            
        print(f"Downloading {key} with 16 parallel connections...")
        if not dry_run:
            archive_name = f"{key}.archive"
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-o", archive_name, url], check=True)
            
            print(f"Extracting {key}...")
            if url.endswith(".zip"):
                subprocess.run(["unzip", "-q", archive_name], check=True)
            else:
                subprocess.run(["tar", "-xf", archive_name], check=True)
            
            # Delete the archive after extraction to save disk space
            os.remove(archive_name)
        else:
            print(f"[DRY RUN] Would download and extract {key}")


def get_audio_file_length_in_seconds(wav_file_path: str) -> float:
    """
    Reads a .wav file and tells us how many seconds long it is.
    """
    if not os.path.exists(wav_file_path):
        return 0.0
        
    try:
        with wave.open(wav_file_path, 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as error:
        print(f"Warning: Could not read duration for {wav_file_path}: {error}")
        return 0.0


def create_the_audio_map_manifest(audio_folder: str, transcripts: dict, dry_run: bool = False):
    """
    Creates the 'train_manifest.json' file.
    """
    print(f"Creating the Audio Map (Manifest) for folder: {audio_folder}...")
    output_file = "train_manifest.json"
    
    if dry_run:
        print(f"[DRY RUN] Would create {output_file} from {audio_folder}")
        return

    audio_dir = Path(audio_folder)
    manifest_entries = []
    
    # Iterate through all wav files in the directory
    for audio_file in audio_dir.glob("**/*.wav"):
        file_id = audio_file.stem
        
        # We only add the file if we have a transcript for it
        if file_id in transcripts:
            entry = {
                "audio_filepath": str(audio_file.absolute()),
                "duration": get_audio_file_length_in_seconds(str(audio_file)),
                "text": transcripts[file_id]
            }
            manifest_entries.append(entry)
        
    # Write all entries as JSON Lines
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Manifest created with {len(manifest_entries)} entries.")


def create_lora_settings_for_low_memory() -> str:
    """
    Writes a temporary YAML file containing the LoRA (PEFT) rules.
    """
    lora_config_path = "lora_settings.yaml"
    
    config = {
        "peft": {
            "peft_type": "lora",
            "target_modules": ["linear_q", "linear_k", "linear_v", "linear_out"],
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05
        }
    }
    
    with open(lora_config_path, 'w') as f:
        yaml.dump(config, f)
        
    return lora_config_path


def print_the_final_command_to_start_training(lora_config_file: str):
    """
    Prints the exact Python command needed to run the official NVIDIA script.
    """
    print("\n" + "="*50)
    print("FINAL STEP: RUN THIS COMMAND IN COLAB TO START TRAINING")
    print("="*50)
    
    command = (
        f"python {OFFICIAL_SCRIPT_NAME} \\\n"
        f"    --config-path='.' \\\n"
        f"    --config-name='{OFFICIAL_CONFIG_NAME}' \\\n"
        f"    +init_from_pretrained_model='nvidia/parakeet-tdt-1.1b' \\\n"
        f"    +model.peft.peft_type='lora' \\\n"
        f"    +model.peft.lora_cfg.target_modules='[linear_q, linear_k, linear_v, linear_out]' \\\n"
        f"    model.train_ds.manifest_filepath='train_manifest.json' \\\n"
        f"    model.train_ds.is_tarred=false \\\n"
        f"    model.train_ds.use_lhotse=false \\\n"
        f"    model.train_ds.max_duration=1000.0 \\\n"
        f"    model.train_ds.min_duration=0.1 \\\n"
        f"    trainer.precision='16-mixed' \\\n"
        f"    trainer.accelerator='gpu' \\\n"
        f"    trainer.devices=1 \\\n"
        f"    trainer.max_epochs=5"
    )
    
    print(command)
    print("="*50 + "\n")


def main():
    """
    The main flow to prepare your Google Colab for training.
    """
    parser = argparse.ArgumentParser(description="Manager for Parakeet-TDT 1.1b Fine-Tuning")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args(args = [])

    print(">>> PARAKEET-TDT COLAB MANAGER STARTING <<<")
    
    # 1. Get official files
    download_official_nvidia_files(dry_run=args.dry_run)
    
    # 2. Get audio data
    download_and_extract_speech_data(dry_run=args.dry_run)
    
    # 3. Create the manifest (map)
    example_transcripts = {"sample": "this is an example transcript"}
    create_the_audio_map_manifest(
        audio_folder="LibriSpeech/dev-clean", 
        transcripts=example_transcripts,
        dry_run=args.dry_run
    )
    
    # 4. Prepare LoRA rules
    lora_file = create_lora_settings_for_low_memory()
    
    # 5. Show the start command
    print_the_final_command_to_start_training(lora_file)
    
    print(">>> PREPARATION FINISHED. YOU ARE READY TO TRAIN. <<<")

if __name__ == '__main__':
    main()
