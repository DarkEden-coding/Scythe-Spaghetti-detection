from synthpipe import GENERATOR_VERSION
from synthpipe.version import code_hash


def test_version_and_source_fingerprint(tmp_path):
    assert GENERATOR_VERSION == '0.3.0'
    files = ['synthetic_data/pipeline/synthpipe/a.py',
             'synthetic_data/pipeline/synthpipe/nested/b.py',
             'synthetic_data/pipeline/synthpipe/profiles/printer/config.json',
             'synthetic_data/blender/scenegen/a.py']
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    original = code_hash(repo_root=tmp_path)
    assert len(original) == 64
    assert code_hash(short=True, repo_root=tmp_path) == original[:16]
    for name in files:
        path = tmp_path/name
        value = path.read_text()
        path.write_text(value+'change')
        assert code_hash(repo_root=tmp_path) != original
        path.write_text(value)
    for name in ('synthetic_data/blender/__pycache__/ignored.py',
                 'synthetic_data/pipeline/synthpipe/__pycache__/ignored.py',
                 'synthetic_data/pipeline/synthpipe/ignored.txt'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('ignored')
    assert code_hash(repo_root=tmp_path) == original
    path = tmp_path/files[0]
    path.rename(path.with_name('renamed.py'))
    assert code_hash(repo_root=tmp_path) != original
