from pathlib import Path

from dump_py_to_text import iter_dumpable_files


def test_iter_dumpable_files_includes_python_and_markdown(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# notes\n", encoding="utf-8")
    ignored_dir = tmp_path / "__pycache__"
    ignored_dir.mkdir()
    (ignored_dir / "ghost.py").write_text("print('ghost')\n", encoding="utf-8")

    files = iter_dumpable_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "alpha.py",
        "notes.md",
    ]
