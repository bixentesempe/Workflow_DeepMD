from __future__ import annotations

from pathlib import Path
from typing import List, Sequence


def _as_path_list(paths: Sequence[str | Path]) -> List[Path]:
    return [Path(p).expanduser().resolve() for p in paths]


def discover_structure_files(
    roots: Sequence[str | Path],
    patterns: Sequence[str] | str | None = None,
    recursive: bool = True,
) -> List[Path]:
    """
    Discover structure files from one or many input roots.

    Parameters
    ----------
    roots:
        Directories or explicit file paths.
    patterns:
        Glob patterns used inside directories. If None, uses a generic set
        covering common VASP, XYZ, and ASE trajectory formats.
    recursive:
        If True, search recursively within directories.
    """
    if patterns is None:
        patterns = ["vasprun*.xml", "XDATCAR*", "*.xyz", "*.extxyz", "*.traj"]
    if isinstance(patterns, str):
        patterns = [patterns]
    files: List[Path] = []
    for root in _as_path_list(roots):
        if root.is_file():
            if any(root.match(pat) for pat in patterns):
                files.append(root)
            continue
        if not root.exists():
            raise FileNotFoundError(f"Input root does not exist: {root}")
        globber = root.rglob if recursive else root.glob
        for pat in patterns:
            files.extend([p.resolve() for p in globber(pat) if p.is_file()])

    unique = sorted({str(p): p for p in files}.values(), key=lambda p: str(p))
    if not unique:
        raise FileNotFoundError(f"No structure files found under: {roots}")
    return unique


def ensure_directory(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out
