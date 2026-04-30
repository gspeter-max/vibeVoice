import os
import sys
import subprocess
import pytest
from pathlib import Path

def test_colab_training_pipeline_syntax():
    """
    Test that the colab_training_pipeline.py script has valid Python syntax.
    This ensures that the script we provide to the user for Google Colab
    does not have basic indentation or syntax errors before they attempt to run it.
    """
    script_path = Path("scripts/colab_training_pipeline.py")
    
    assert script_path.exists(), "The training pipeline script must exist."
    
    # Run py_compile to check syntax without executing the code
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(script_path)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Syntax error in script:\n{result.stderr}"

def test_colab_training_pipeline_imports_and_functions():
    """
    Test that the script can be imported and its functions inspected
    without actually executing the training loop (which would require a GPU
    and massive amounts of RAM).
    """
    import scripts.colab_training_pipeline as pipeline
    
    # Verify the functions are defined and return the expected structures
    peft_cfg = pipeline.configure_lora_adapters()
    assert "peft" in peft_cfg
    assert peft_cfg["peft"]["lora_rank"] == 8
    assert "linear_q" in peft_cfg["peft"]["target_modules"]
    
    aug_cfg = pipeline.add_noise_rules_to_model()
    assert len(aug_cfg) == 3
    assert any(rule["type"] == "white_noise" for rule in aug_cfg)
    assert any(rule["type"] == "impulse_response" for rule in aug_cfg)
    assert any(rule["type"] == "noise" for rule in aug_cfg)
