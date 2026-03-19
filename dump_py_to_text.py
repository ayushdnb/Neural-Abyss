from __future__ import annotations

from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "codes" / "code_dump.txt"
APPEND = False
INCLUDE_SUFFIXES = {".md", ".py"}
IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    "env",
    "results",
    "venv",
}


def should_ignore(path: Path) -> bool:
    """Return whether any path segment is excluded from the dump."""
    return any(part in IGNORE_DIRS for part in path.parts)


def iter_source_files(base_dir: Path) -> list[Path]:
    """Collect Python and Markdown files under ``base_dir``."""
    files: list[Path] = []
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(base_dir)
        if should_ignore(rel_path) or path.suffix.lower() not in INCLUDE_SUFFIXES:
            continue
        files.append(path)
    files.sort(key=lambda path: str(path.relative_to(base_dir)).lower())
    return files


def main() -> None:
    """Write a deterministic repository source dump."""
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    source_files = iter_source_files(BASE_DIR)
    mode = "a" if APPEND else "w"

    with OUT_FILE.open(mode, encoding="utf-8", errors="replace") as output:
        if not APPEND:
            output.write(
                "# Aggregated repository sources\n"
                f"# Base: {BASE_DIR}\n"
                f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                f"# Total files: {len(source_files)}\n"
                f"{'-' * 80}\n"
            )

        for index, path in enumerate(source_files, start=1):
            rel_path = path.relative_to(BASE_DIR)
            header = (
                f"\n\n====[ {index}/{len(source_files)} | {rel_path} ]"
                f"{'=' * max(1, 78 - len(str(rel_path)))}\n"
            )
            output.write(header)
            try:
                output.write(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                output.write(f"# [ERROR READING FILE: {exc}]\n")
            output.write(f"\n====[ END {rel_path} ]{'=' * 60}\n")

    print(f"Wrote {len(source_files)} files to {OUT_FILE}")


if __name__ == "__main__":
    main()
