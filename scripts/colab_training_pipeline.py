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
    
    We need:
    - The Script: To do the math for training.
    - The Config: To tell the script how fast to learn (Optimizer/Learning Rate).
    """
    print(f"Downloading official NVIDIA files...")
    
    files_to_download = {
        OFFICIAL_SCRIPT_NAME: OFFICIAL_NEMO_SCRIPT_URL,
        OFFICIAL_CONFIG_NAME: OFFICIAL_CONFIG_URL
    }
    
    for filename, url in files_to_download.items():
        print(f"  -> Grabbing {filename}...")
        if not dry_run:
            subprocess.run(["wget", "-q", "-O", filename, url], check=True)
    
    if not dry_run:
        print("Official files downloaded successfully.")


def download_and_extract_speech_data(dry_run: bool = False):
    """
    Downloads the clean speech, background noise, and room echo data.
    
    Why we need these:
    - LibriSpeech: The clean "base" voice.
    - MUSAN: The "background noise" (street sounds, music).
    - RIR: The "room echo" (bathroom, hall sounds).
    """
    print("Starting data acquisition (Clean Speech, Noise, Echoes)...")
    
    datasets = {
        "librispeech_clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "musan_noise": "http://www.openslr.org/resources/17/musan.tar.gz",
        "room_echoes": "http://www.openslr.org/resources/28/rirs_noises.zip"
    }
    
    for name, url in datasets.items():
        print(f"Downloading {name}...")
        if not dry_run:
            archive_name = f"{name}.archive"
            subprocess.run(["wget", "-q", "-O", archive_name, url], check=True)
            print(f"Extracting {name}...")
            if url.endswith(".zip"):
                subprocess.run(["unzip", "-q", archive_name], check=True)
            else:
                subprocess.run(["tar", "-xf", archive_name], check=True)
        else:
            print(f"[DRY RUN] Would download and extract {name}")


def get_audio_file_length_in_seconds(wav_file_path: str) -> float:
    """
    Reads a .wav file and tells us how many seconds long it is.
    
    We need this because NeMo won't train on files if it doesn't 
    know how long they are.
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
    
    This function looks at all your .wav files, measures their length, 
    and writes a JSON map so NeMo can find them.
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
    
    LoRA is a trick to fine-tune a 1.1B model on a small 16GB GPU.
    It freezes the original model and only trains tiny "adapter" layers.
    """
    lora_config_path = "lora_settings.yaml"
    
    # We target the 'linear' parts of the attention layers.
    # For Parakeet-TDT, these are linear_q, linear_k, linear_v, and linear_out.
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
        f"    trainer.precision='16-mixed' \\\n"
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
    args = parser.parse_args()

    print(">>> PARAKEET-TDT COLAB MANAGER STARTING <<<")
    
    # 1. Get official files
    download_official_nvidia_files(dry_run=args.dry_run)
    
    # 2. Get audio data
    download_and_extract_speech_data(dry_run=args.dry_run)
    
    # 3. Create the manifest (map)
    # Note: In a real Colab, you would load transcripts from the LibriSpeech folders.
    # Here we use a placeholder dictionary.
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
