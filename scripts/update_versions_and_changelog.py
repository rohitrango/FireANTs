import datetime
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
FIREANTS_PYPROJECT = ROOT_DIR / "pyproject.toml"
FUSEDOPS_PYPROJECT = ROOT_DIR / "fused_ops" / "pyproject.toml"
CHANGELOG = ROOT_DIR / "CHANGELOG.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def parse_version_from_pyproject(pyproject_text: str) -> str:
    for line in pyproject_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            # Expect format: version = "x.y.z"
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("Could not find version in pyproject.toml")


def bump_version(version: str, part: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown version part: {part}")
    return f"{major}.{minor}.{patch}"


def update_pyproject_version(path: Path, part: Optional[str] = None) -> tuple[str, str]:
    """
    Update the version in the given pyproject.toml.

    Returns (old_version, new_version). If part is None, no change is made and
    (current_version, current_version) is returned.
    """
    text = read_text(path)
    current = parse_version_from_pyproject(text)

    if part is None:
        return current, current

    new_version = bump_version(current, part)

    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            prefix, _ = line.split("=", 1)
            prefix = prefix.rstrip()
            lines[i] = f'{prefix} = "{new_version}"'
            break

    write_text(path, "\n".join(lines) + ("\n" if text.endswith("\n") else ""))
    return current, new_version


def prompt_version_choice(label: str) -> Optional[str]:
    while True:
        choice = input(
            f"Select {label} version bump [major/minor/patch/skip]: "
        ).strip().lower()
        if choice in {"major", "minor", "patch"}:
            return choice
        if choice == "skip":
            return None
        print("Invalid choice. Please enter 'major', 'minor', 'patch', or 'skip'.")


def collect_changelog_lines() -> list[str]:
    print("Enter changelog lines (Markdown).")
    print("Press ENTER on an empty line when you are done.\n")
    lines: list[str] = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return lines


def build_changelog_entry(
    fireants_version: str, fusedops_version: str, body_lines: list[str]
) -> str:
    today = datetime.date.today().isoformat()
    title = (
        f"### {today} - fireants {fireants_version}, "
        f"fireants_fused_ops {fusedops_version} - changes"
    )
    entry_lines = [title, ""]
    entry_lines.extend(body_lines)
    entry_lines.append("")  # trailing blank line
    return "\n".join(entry_lines)


def update_changelog(
    fireants_version: str, fusedops_version: str, body_lines: list[str]
) -> None:
    if not body_lines:
        print("No changelog lines entered, skipping CHANGELOG.md update.")
        return

    existing = read_text(CHANGELOG)
    new_entry = build_changelog_entry(fireants_version, fusedops_version, body_lines)

    # Insert new entry after the initial header and intro paragraph (if present).
    lines = existing.splitlines()

    insert_index = len(lines)
    # We keep the initial "# Changelog" header and the following explanatory text,
    # then insert our new section after the first blank line following that block.
    if lines and lines[0].lstrip().startswith("# Changelog"):
        for i in range(1, len(lines)):
            if lines[i].strip() == "" and i >= 2:
                insert_index = i + 1
                break

    new_lines = []
    new_lines.extend(lines[:insert_index])
    if new_lines and new_lines[-1].strip() != "":
        new_lines.append("")
    new_lines.append(new_entry)
    if insert_index < len(lines):
        if lines[insert_index].strip() != "":
            new_lines.append("")
        new_lines.extend(lines[insert_index:])

    write_text(CHANGELOG, "\n".join(new_lines) + ("\n" if existing.endswith("\n") else ""))


def main() -> None:
    print("=== FireANTs / fused_ops version & changelog helper ===")

    # Fireants version
    fireants_choice = prompt_version_choice("fireants")
    _, fireants_new = update_pyproject_version(FIREANTS_PYPROJECT, fireants_choice)

    # fused_ops version
    fusedops_choice = prompt_version_choice("fireants_fused_ops")
    _, fusedops_new = update_pyproject_version(FUSEDOPS_PYPROJECT, fusedops_choice)

    # Changelog
    changelog_lines = collect_changelog_lines()
    update_changelog(fireants_new, fusedops_new, changelog_lines)

    print("Done.")


if __name__ == "__main__":
    main()

