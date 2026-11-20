"""Regression test: CLI tools and scripts must be copied into /app in the Docker runtime image.

Checks the Dockerfile text (no Docker daemon needed) and that every source path it copies exists.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"

# (source path inside /build of the base stage, destination under /app)
EXPECTED_COPIES = [
    ("/build/cli", "/app/cli"),
    ("/build/fireants/scripts", "/app/scripts"),
]

COPY_FROM_BASE = re.compile(r"^COPY\s+--from=base\s+(\S+)\s+(\S+)\s*$", re.MULTILINE)


def _runtime_stage() -> str:
    text = DOCKERFILE.read_text()
    # Everything after the last FROM line is the runtime stage
    return text[text.rfind("\nFROM ") + 1:]


def test_dockerfile_exists():
    assert DOCKERFILE.is_file()


@pytest.mark.parametrize("src, dst", EXPECTED_COPIES)
def test_runtime_stage_copies_into_app(src, dst):
    stage = _runtime_stage()
    copies = COPY_FROM_BASE.findall(stage)
    assert (src, dst) in copies, f"runtime stage must contain 'COPY --from=base {src} {dst}'"


def test_copies_happen_after_workdir_app():
    stage = _runtime_stage()
    workdir_pos = stage.index("WORKDIR /app")
    for _, dst in EXPECTED_COPIES:
        pos = stage.index(f" {dst}")
        assert pos > workdir_pos, f"{dst} copy must come after WORKDIR /app"


@pytest.mark.parametrize("src, _dst", EXPECTED_COPIES)
def test_copy_sources_exist_in_repo(src, _dst):
    rel = src[len("/build/"):]
    assert (REPO_ROOT / rel).is_dir(), f"{rel} must exist in the repo to be copied"


def test_template_builder_and_configs_are_present():
    template_dir = REPO_ROOT / "fireants" / "scripts" / "template"
    assert (template_dir / "build_template.py").is_file()
    configs = sorted((template_dir / "configs").glob("*.yaml"))
    assert configs, "template builder should ship at least one YAML config"
