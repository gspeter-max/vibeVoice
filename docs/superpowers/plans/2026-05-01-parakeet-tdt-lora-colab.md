# Parakeet-TDT 1.1b PEFT Colab Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 
> **READ BEFORE STARTING:**
> - Read `/Users/apple/.gemini/GEMINI.md`
> - Follow `## Mind Set rules` throughout execution
> - You are writing a script meant to be run in a resource-constrained Google Colab environment (16GB VRAM).

## Goal information for freshAgent

- **The Objective:** Create a robust fine-tuning pipeline for the `nvidia/parakeet-tdt-1.1b` ASR model targeting faint and noisy audio, specifically designed to run on a Google Colab T4 GPU (16GB VRAM).
- **The Constraints:** Because the model is 1.1 Billion parameters, full fine-tuning will immediately crash Colab. We MUST use PEFT (Parameter-Efficient Fine-Tuning) via LoRA.
- **The Data:** We are using a "Mini-Dataset" approach to save disk space and RAM.
    - Base Speech: LibriSpeech `dev-clean` (SLR12)
    - Noise: A tiny subset of MUSAN (SLR17)
    - Acoustics: A tiny subset of RIR (SLR28)
- **The Trick:** Instead of pre-mixing audio, we use NVIDIA NeMo's dynamic YAML augmentors to inject Gain (volume changes), RIR (echo), and MUSAN (noise) on-the-fly during training.
- **How to think about the code:** Do not just write a massive script. Think about the Colab environment. You need a setup phase (installing libraries), a data prep phase (downloading minimal data and writing JSON manifests), a configuration phase (defining LoRA and Augmentors), and an execution phase.
- **Where to look:** Review `docs/superpowers/plans/` for formatting, but your primary output will be a well-documented Python script (e.g., `scripts/colab_finetune_parakeet.py`) that the user can copy into Colab.

## Architecture

The script will be structured chronologically for a Colab notebook:
1. **Environment Setup:** Commands to install `nemo_toolkit['asr']` and `peft`.
2. **Data Acquisition:** Logic to `wget` and extract only `dev-clean` from LibriSpeech, and small slices of MUSAN/RIR.
3. **Manifest Generation:** Functions to read the extracted audio and write `.json` NeMo manifests (requiring `audio_filepath`, `duration`, `text`).
4. **Model Instantiation:** Loading the frozen base 1.1B model.
5. **PEFT Configuration:** Injecting LoRA adapters into the FastConformer's attention layers (`linear_q`, `linear_k`, `linear_v`, `linear_out`) with rank=8.
6. **Data Loader Configuration:** Defining the `augmentor` block to apply random gain reduction, RIR convolution, and MUSAN mixing.
7. **Trainer Execution:** Using PyTorch Lightning to train the adapters in `16-mixed` precision.
8. **Checkpointing:** Logic to save the tiny LoRA `.nemo` file to Google Drive.

## Important Rule to follow

- **CRITICAL:** Add detailed docs in functions and explain the code and logic in comments.
- **CRITICAL:** Make the code function names and variable names clear and literal. Do not use abbreviations. A 5-year-old child should easily understand what the function does (e.g., use `download_and_extract_speech_data` instead of `get_data`, use `add_noise_rules_to_model` instead of `config_aug`).
- **Write code so a developer gets the highest speed to read the code.**
- **Explain like a fresher.**
- **Write docs in your step-by-step simple style.**
- **Make the docs in function and file headers human-readable and literal.**

---

## Task Structure

### Task 1 : [ read out instruction file ]
- read GEMINI.md file
- **CRITICAL:** add detailed docs in functions and explain the code and logic in comments.
- (**CRITICAL** make the code function name and variable name clear not easily to understand instand of short and confusing names)
- avoid surface level ( happy path ) tests use detailed tests.
- write code function name and docs and code like this **developer get hightest speed to read the code**

### Task 2: Initialize Workspace and TDD for Manifest Generation

**Files:**
- Create: `src/data_record/colab_manifest_builder.py`
- Test: `tests/test_colab_manifest_builder.py`

**Instructions for the Agent:**
Instead of testing the massive NeMo training loop locally, test the data preparation logic. Write functions that can take a mock directory of `.wav` files and a mock transcript file, and correctly output a NeMo `.json` manifest.

- [ ] **Step 1: Write the failing test**
  Think: How do we ensure the manifest builder correctly calculates audio duration and formats the JSON string? Write a test that creates dummy `.wav` files, passes the directory to your builder function, and asserts the output JSON contains the correct keys (`audio_filepath`, `duration`, `text`).

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_colab_manifest_builder.py -v`
  Expected: FAIL (module/function not found)

- [ ] **Step 3: Write minimal implementation**
  Think: Use standard Python `json` and a lightweight audio library (like `wave` or `soundfile` which are built-in) to read duration. Write literal, highly-documented code in `src/data_record/colab_manifest_builder.py`.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_colab_manifest_builder.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add src/data_record/colab_manifest_builder.py tests/test_colab_manifest_builder.py
  git commit -m "feat: add manifest builder logic for colab pipeline"
  ```

### Task 3: Draft and Validate the Colab Training Script Structure

**Files:**
- Create: `scripts/colab_training_pipeline.py`
- Test: `tests/test_colab_training_pipeline.py`

**Instructions for the Agent:**
This file will be the final product the user runs in Colab. While we cannot run the full GPU training loop locally, we MUST validate that the script is syntactically correct and its core configurations (like PEFT and Augmentors) are structurally sound. To do this, design the script with functions returning configurations, rather than a single monolithic block, so they can be unit-tested.

- [ ] **Step 1: Write the failing tests for Configurations**
  Think: Write tests in `test_colab_training_pipeline.py` that import the configuration functions from the script and assert the dictionaries contain the right keys (e.g., `peft_type: lora`, `target_modules` for FastConformer, `augmentor` dict with `gain` and `impulse_response`).

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_colab_training_pipeline.py -v`
  Expected: FAIL (module/function not found)

- [ ] **Step 3: Write the Data Download and Config Sections**
  Think: In `scripts/colab_training_pipeline.py`, write Python code utilizing `os.system` or `subprocess` to download SLR12 (dev-clean), SLR17 (MUSAN), and SLR28 (RIR). Define functions like `get_lora_config()` and `get_augmentor_config()` returning the dictionaries. 

- [ ] **Step 4: Write the Trainer and Google Drive Save Logic**
  Think: Initialize the PyTorch Lightning trainer with `precision="16-mixed"` (crucial for Colab). Write the logic to save the model to `/content/drive/MyDrive/`. Wrap this in a `if __name__ == "__main__":` block so importing for tests doesn't trigger execution.

- [ ] **Step 5: Run tests and Verify Syntax**
  Run: `pytest tests/test_colab_training_pipeline.py -v` and `python -m py_compile scripts/colab_training_pipeline.py`
  Expected: PASS and Silent exit.

- [ ] **Step 6: Commit**
  ```bash
  git add scripts/colab_training_pipeline.py tests/test_colab_training_pipeline.py
  git commit -m "feat: draft and validate colab pipeline configuration logic"
  ```

## Self-Review (before sharing the plan)
- [x] run sub-agent for reveiw the plan.
- [x] Tests catch real edge cases, not just happy paths? Yes, the mock audio tests will ensure robust JSON output regardless of input format constraints.