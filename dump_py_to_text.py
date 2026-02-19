from pathlib import Path
from datetime import datetime

# --- Config ---
BASE_DIR = Path(r"C:\Kishan\RL_Project\rich_feature")
OUT_FILE = BASE_DIR / "codes" / "rich_feature.txt"

# Set to True if you want to APPEND on every run (instead of recreating fresh)
APPEND = False

# Folders to ignore anywhere in the path
IGNORE_DIRS = {
    "__pycache__", ".git", ".hg", ".svn",
    "venv", ".venv", "env", ".env",
    ".mypy_cache", ".pytest_cache", "build", "dist"
}

def should_ignore(p: Path) -> bool:
    """Return True if any part of the path is in IGNORE_DIRS."""
    return any(part in IGNORE_DIRS for part in p.parts)

def main():
    # Ensure output directory exists
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Gather all .py files (recursively), excluding ignored dirs
    py_files = []
    for p in BASE_DIR.rglob("*.py"):
        if p.is_file() and not should_ignore(p.relative_to(BASE_DIR)):
            py_files.append(p)

    # Sort deterministically by relative path
    py_files.sort(key=lambda p: str(p.relative_to(BASE_DIR)).lower())

    mode = "a" if APPEND else "w"
    with OUT_FILE.open(mode, encoding="utf-8", errors="replace") as out:
        if not APPEND:
            out.write(
                f"# Aggregated Python sources\n"
                f"# Base: {BASE_DIR}\n"
                f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                f"# Total files: {len(py_files)}\n"
                f"{'-'*80}\n"
            )

        for idx, f in enumerate(py_files, 1):
            rel = f.relative_to(BASE_DIR)
            header = (
                f"\n\n====[ {idx}/{len(py_files)} | {rel} ]"
                f"{'=' * max(1, 78 - len(str(rel)))}\n"
            )
            out.write(header)
            try:
                code = f.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                code = f"# [ERROR READING FILE: {e}]\n"
            out.write(code)
            out.write(f"\n====[ END {rel} ]{'=' * 60}\n")

    print(f"Done. Wrote {len(py_files)} .py files into:\n{OUT_FILE}")

if __name__ == "__main__":
    main()