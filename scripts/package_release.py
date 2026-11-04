"""Create a source archive from Git-visible files, excluding runtime artifacts."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "data",
    "logs",
    "outputs",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".part",
    ".log",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".key",
    ".pem",
}


def package(output: Path) -> int:
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    files = []
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        if EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        if (
            path.suffix in EXCLUDED_SUFFIXES
            or path.name.startswith(".env")
            and path.name != ".env.example"
        ):
            continue
        if path.is_file() and not path.is_symlink() and path.resolve() != output.resolve():
            files.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=Path("EquiRouter") / path.relative_to(ROOT), recursive=False)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/EquiRouter-source.tar.gz"))
    args = parser.parse_args()
    print(f"Archived {package(args.output)} source files to {args.output}")


if __name__ == "__main__":
    main()
