from pathlib import Path

BASE_DIR = Path(r"C:\Kishan\RL_Project\rich_feature\Infinite_War_Simulation")

IGNORE_DIRS = {
    "__pycache__", ".git", ".hg", ".svn",
    "venv", ".venv", "env", ".env",
    ".mypy_cache", ".pytest_cache", "build", "dist"
}

def should_ignore(p: Path) -> bool:
    return any(part in IGNORE_DIRS for part in p.parts)

py_files = sorted(
    p.relative_to(BASE_DIR)
    for p in BASE_DIR.rglob("*.py")
    if p.is_file() and not should_ignore(p.relative_to(BASE_DIR))
)

print("COUNT:", len(py_files))
for p in py_files:
    print(p)