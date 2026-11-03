"""Read-only run health and throughput report."""
import json
import shutil
import time
from synthpipe.manifest import Manifest
from synthpipe.sampler import run_dir


def add_arguments(parser):
    parser.add_argument('--run', required=True)


def run(args):
    directory = run_dir(args.run)
    if not (directory / 'manifest.sqlite').is_file():
        raise ValueError('No manifest for run '+args.run)
    with Manifest(directory / 'manifest.sqlite') as manifest:
        counts = manifest.counts()
        now = time.time()
        recent = {minutes: manifest.db.execute(
            "SELECT COUNT(*) FROM jobs WHERE state='done' AND updated>=?", 
            (now-minutes*60,)).fetchone()[0] for minutes in (15, 60)}
        rates = {minutes: count*60/minutes for minutes, count in recent.items()}
        crash_attempts = manifest.db.execute('SELECT COALESCE(SUM(crash_attempts),0) FROM jobs').fetchone()[0]
        averages = manifest.db.execute('SELECT AVG(render_seconds),AVG(post_seconds) FROM jobs').fetchone()
        errors = [dict(row) for row in manifest.db.execute('''SELECT job_id,state,attempts,error
                  FROM jobs WHERE error IS NOT NULL ORDER BY updated DESC LIMIT 10''')]
    remaining = sum(counts[s] for s in ('pending', 'rendering', 'rendered', 'postprocessing'))
    disk = shutil.disk_usage(directory)
    run_bytes = 0
    for path in directory.rglob('*'):
        try:
            if path.is_file():
                run_bytes += path.stat().st_size
        except FileNotFoundError:
            pass  # A live worker renamed or removed an output during the scan.
    print(json.dumps(dict(run=args.run, counts=counts, jobs_per_hour=rates[60],
        jobs_done_last_15m=recent[15], jobs_done_last_60m=recent[60],
        jobs_per_hour_15m=rates[15], jobs_per_hour_60m=rates[60],
        eta_hours=remaining/rates[15] if rates[15] else None,
        crash_attempts=crash_attempts,
        avg_render_seconds=averages[0], avg_post_seconds=averages[1], failures=errors,
        disk=dict(run_bytes=run_bytes, free_bytes=disk.free, total_bytes=disk.total),
        stopped=(directory/'STOP').exists(), paused=(directory/'.PAUSE_CLAIMS').exists()), indent=2))
    return 0
