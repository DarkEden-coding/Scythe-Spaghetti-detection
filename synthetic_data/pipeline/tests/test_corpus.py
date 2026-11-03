import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from synthpipe import corpus


@pytest.mark.parametrize('license', [
    'BSD License', 'CC0', 'CC-BY-4.0', 'CC-BY-SA-4.0', 'Public Domain', 'GNU - GPL', 'GNU - LGPL',
    'Creative Commons - Attribution', 'Creative Commons - Attribution - Share Alike',
    'Creative Commons - Public Domain Dedication',
])
def test_license_allowlist(license):
    assert corpus.permissive_license(license) == license


@pytest.mark.parametrize('license', [None, '', 'unknown_license', 'CC-BY-NC', 'CC-BY-ND',
    'Creative Commons - Attribution - Non-Commercial',
    'Attribution - Non-Commercial - Share Alike', 'Creative Commons - Attribution - No Derivatives',
    'All rights reserved', 'Creative Commons - Unrecognized License'])
def test_restricted_unknown_license(license):
    assert corpus.permissive_license(license) is None


def test_mesh_geometry_and_base_is_not_overhang():
    box = trimesh.creation.box([10, 10, 5])
    details = corpus.inspect_mesh(box)
    assert details['bbox_mm'] == [10, 10, 5]
    assert details['faces'] == 12
    assert not details['has_overhangs']
    open_mesh = box.copy()
    open_mesh.update_faces(np.arange(len(box.faces) - 1))
    with pytest.raises(ValueError, match='not_closed'):
        corpus.inspect_mesh(open_mesh)
    with pytest.raises(ValueError, match='aspect_ratio'):
        corpus.inspect_mesh(trimesh.creation.box([1, 1, 1000]))
    with pytest.raises(ValueError, match='single_body'):
        corpus.inspect_mesh(trimesh.util.concatenate([box, box.copy().apply_translation([30, 0, 0])]))


def test_overhang_excludes_bottom_but_detects_elevated_downward_faces():
    # A concave L-shaped solid has an elevated underside.
    polygon = np.array([[0, 0], [2, 0], [2, 2], [5, 2], [5, 3], [0, 3]])
    vertices = np.array([[x, y, z] for y in [0, 2] for x, z in polygon])
    triangles = [[0, 1, 2], [0, 2, 5], [2, 3, 4], [2, 4, 5]]
    faces = [t[::-1] for t in triangles] + [[i+6 for i in t] for t in triangles]
    for i in range(6):
        j = (i + 1) % 6
        faces += [[i, j, j+6], [i, j+6, i+6]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    assert corpus.inspect_mesh(mesh)['has_overhangs']


def row(tmp_path, file_id=1, license='CC0'):
    mesh = trimesh.creation.box([10, 10, 5])
    path = tmp_path / f'{file_id}.npz'
    np.savez(path, vertices=mesh.vertices, facets=mesh.faces)
    return {'file_id': file_id, 'file_path': str(path), 'license': license,
            'closed': True, 'vertex_manifold': True, 'edge_manifold': True,
            'solid': True, 'num_components': 1, 'num_facets': 12, 'tags': ['test']}


def test_build_resumes_and_rebuilds_missing_stl(tmp_path, monkeypatch):
    rows = [row(tmp_path), row(tmp_path, 2, 'CC-BY-NC')]
    monkeypatch.setattr(corpus, 'download_metadata', lambda raw: rows)
    root = tmp_path / 'corpus'
    result = corpus.build_corpus(corpus_dir=root)
    assert len(result) == 1
    path = Path(result[0]['stl'])
    stamp = path.stat().st_mtime_ns
    assert json.loads((root / 'index.json').read_text()) == result
    assert corpus.build_corpus(corpus_dir=root) == result
    assert path.stat().st_mtime_ns == stamp
    path.unlink()
    assert corpus.build_corpus(corpus_dir=root) == result
    assert path.exists()
    assert not corpus.build_corpus(corpus_dir=root, min_overhang=0.1)


def test_metadata_requires_explicit_geometry_flags(tmp_path):
    entry = row(tmp_path)
    assert corpus.metadata_rejection(entry) is None
    del entry['closed']
    assert corpus.metadata_rejection(entry) == 'not_closed_manifold_solid'
