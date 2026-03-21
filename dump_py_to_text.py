"""Aggregate repository source and documentation files into one text dump."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "codes" / "code_dump.txt"
APPEND = False
INCLUDE_GLOBS = ("*.py", "*.md")

# Folders to ignore anywhere in the path.
IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    "venv",
    ".venv",
    "env",
    ".env",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
}


def should_ignore(path: Path) -> bool:
    """Return ``True`` when any path component should be ignored."""
    return any(part in IGNORE_DIRS for part in path.parts)


def iter_dumpable_files(base_dir: Path) -> list[Path]:
    """Collect Python and Markdown files under ``base_dir`` deterministically."""
    files: list[Path] = []
    seen: set[Path] = set()

    for pattern in INCLUDE_GLOBS:
        for path in base_dir.rglob(pattern):
            if not path.is_file():
                continue
            rel_path = path.relative_to(base_dir)
            if should_ignore(rel_path):
                continue
            if path in seen:
                continue
            seen.add(path)
            files.append(path)

    files.sort(key=lambda path: str(path.relative_to(base_dir)).lower())
    return files


def main() -> None:
    """Write the aggregated repository dump to ``codes/code_dump.txt``."""
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    source_files = iter_dumpable_files(BASE_DIR)

    mode = "a" if APPEND else "w"
    with OUT_FILE.open(mode, encoding="utf-8", errors="replace") as out:
        if not APPEND:
            out.write(
                f"# Aggregated repository sources\n"
                f"# Base: {BASE_DIR}\n"
                f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                f"# Total files: {len(source_files)}\n"
                f"{'-' * 80}\n"
            )

        for idx, path in enumerate(source_files, 1):
            rel_path = path.relative_to(BASE_DIR)
            header = (
                f"\n\n====[ {idx}/{len(source_files)} | {rel_path} ]"
                f"{'=' * max(1, 78 - len(str(rel_path)))}\n"
            )
            out.write(header)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                content = f"# [ERROR READING FILE: {exc}]\n"
            out.write(content)
            out.write(f"\n====[ END {rel_path} ]{'=' * 60}\n")

    print(f"Done. Wrote {len(source_files)} files into:\n{OUT_FILE}")


if __name__ == "__main__":
    main()
