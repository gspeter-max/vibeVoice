import subprocess
import os

def test_start_sh_dry_run_no_refiner_output():
    """
    Ensure that start.sh no longer mentions the transcript refiner
    or llama-server in its banner/config output, even in dry-run mode.
    """
    env = os.environ.copy()
    env["START_SH_DRY_RUN"] = "1"
    env["RECORDING_MODE"] = "silence_streaming"
    
    # We explicitly unset these to ensure we aren't getting them from the environment
    env.pop("TEXT_REFINER_ENABLED", None)
    
    result = subprocess.run(
        ["./start.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=True
    )
    
    output = result.stdout
    
    # Negative assertions: these should NOT be in the output anymore
    assert "Refiner" not in output
    assert "LLAMA_PORT" not in output
    assert "TEXT_REFINER" not in output
    assert "llama-server" not in output
    assert "Qwen2.5" not in output

    # Positive assertions: core info should still be there
    assert "Backend" in output
    assert "Mode" in output
    assert "silence_streaming" in output
    assert "Dry run" in output


def test_start_sh_dry_run_prints_streaming_telemetry_banner_when_enabled():
    env = os.environ.copy()
    env["START_SH_DRY_RUN"] = "1"
    env["RECORDING_MODE"] = "silence_streaming"
    env["STREAMING_TELEMETRY_ENABLED"] = "1"

    result = subprocess.run(
        ["./start.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Telemetry" in result.stdout
    assert "enabled" in result.stdout
