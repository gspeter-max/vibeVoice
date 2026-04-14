import os
import subprocess
from pathlib import Path


def test_start_sh_dry_run_warns_and_disables_refiner_when_groq_key_missing():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "START_SH_DRY_RUN": "1",
            "RECORDING_MODE": "silence_streaming",
            "GROQ_API_KEY": "",
        }
    )

    result = subprocess.run(
        [str(repo_root / "start.sh")],
        env=env,
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Refiner  : disabled" in result.stdout
    assert "GROQ_API_KEY is missing" in result.stdout
    assert "llama-server" not in result.stdout
