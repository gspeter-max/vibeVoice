import tomllib
from pathlib import Path


def test_sherpa_onnx_is_in_default_dependencies_without_platform_exclusion():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    dependencies = pyproject["project"]["dependencies"]
    sherpa_dependency = next(
        dependency for dependency in dependencies if dependency.startswith("sherpa-onnx")
    )

    assert sherpa_dependency == "sherpa-onnx>=1.10.0"
