"""Re-export the stable Flatpak's inherited system presets (no GUI required).

Run from the pipeline project: uv run python synthpipe/profiles/export_profiles.py
These upstream OrcaSlicer presets are distributed under its AGPL-3.0 license.
"""
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

SOURCES = {
    'bambu_x1c': ('BBL', 'Bambu Lab X1 Carbon 0.4 nozzle',
                  '0.20mm Standard @BBL X1C', 'Bambu PLA Basic @BBL X1C'),
    'bambu_a1mini': ('BBL', 'Bambu Lab A1 mini 0.4 nozzle',
                     '0.20mm Standard @BBL A1M', 'Generic PLA @BBL A1M'),
    'prusa_mk4': ('Prusa', 'Prusa MK4 0.4 nozzle',
                  '0.20mm Standard @MK4', 'Prusa Generic PLA @MK4'),
    'voron24_300': ('Voron', 'Voron 2.4 300 0.4 nozzle',
                    '0.20mm Standard @Voron', 'Generic PLA @System'),
}


def main():
    work = Path(os.environ.get('SCYTHE_SYNTH_ROOT', '/mnt/Primary/scythe-synth')) / 'gcode_cache' / '.profile-export'
    work.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([
        'flatpak', 'run', '--branch=stable', f'--filesystem={work.resolve()}',
        'com.orcaslicer.OrcaSlicer', '--help'], cwd=work, text=True)
    if not version.startswith('OrcaSlicer-2.4.2:'):
        raise RuntimeError('Expected stable OrcaSlicer 2.4.2')
    archive = subprocess.check_output([
        'flatpak', 'run', '--branch=stable', '--command=tar', 'com.orcaslicer.OrcaSlicer',
        '-C', '/app/share/OrcaSlicer', '-cf', '-', 'profiles'], cwd=work)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        # Flatpak deduplication hard-links many vendor-common presets.
        items = {m.name: json.load(tar.extractfile(m)) for m in tar
                 if (m.isfile() or m.islnk()) and m.name.endswith('.json')}

    def resolve(vendor, kind, name, seen=()):
        candidates = [p for p, d in items.items()
                      if d.get('name') == name and d.get('type') == kind]
        local = [p for p in candidates if p.startswith(f'profiles/{vendor}/')]
        candidates = local or candidates
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous or absent preset {vendor}/{kind}/{name}: {candidates}')
        path = candidates[0]
        if path in seen:
            raise ValueError(f'Inheritance cycle: {path}')
        data = items[path]
        result, lineage = {}, []
        if data.get('inherits'):
            result, lineage = resolve(path.split('/')[1], kind, data['inherits'], (*seen, path))
        result.update(data)
        result.pop('inherits', None)
        result['instantiation'] = 'true'
        return result, [*lineage, path]

    out = Path(__file__).resolve().parent
    origins = {'slicer': 'OrcaSlicer 2.4.2', 'flatpak_branch': 'stable',
               'upstream': 'https://github.com/OrcaSlicer/OrcaSlicer/tree/v2.4.2/resources/profiles',
               'license': 'AGPL-3.0', 'profiles': {}}
    for key, (vendor, machine, process, filament) in SOURCES.items():
        for kind, name in [('machine', machine), ('process', process), ('filament', filament)]:
            data, lineage = resolve(vendor, kind, name)
            filename = f'{key}_{kind}.json'
            (out / filename).write_text(json.dumps(data, indent=2) + '\n')
            origins['profiles'][filename] = {'name': name, 'inheritance': lineage}
    (out / 'sources.json').write_text(json.dumps(origins, indent=2) + '\n')


if __name__ == '__main__':
    main()
