"""Generator identity shared by sampling and run metadata."""
import hashlib
from pathlib import Path

GENERATOR_VERSION = "0.3.0"


def code_hash(short=False, *, repo_root=None):
    """Hash sorted repository-relative source paths and bytes (16 hex if short).

    Length prefixes make path/content boundaries unambiguous. No timestamps,
    absolute paths, bytecode, or generated data enter the fingerprint.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    package = root / 'synthetic_data/pipeline/synthpipe'
    files = set(package.rglob('*.py'))
    files.update(p for p in (package / 'profiles').rglob('*') if p.is_file())
    files.update((root / 'synthetic_data/blender').rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        if '__pycache__' in path.parts:
            continue
        name = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    value = digest.hexdigest()
    return value[:16] if short else value
