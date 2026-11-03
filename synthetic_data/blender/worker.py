"""Persistent Blender worker; stdlib queue implementation intentionally standalone.

--stub also runs under ordinary Python; its .exr files are NumPy placeholders,
not production EXRs. The supervisor uses a matching stub postprocessor.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import threading
import time
import traceback

# Keep these statements in sync with pipeline/synthpipe/manifest.py.
CLAIM_SQL = """UPDATE jobs SET state=?, lease_owner=?, lease_expires=?, updated=?
WHERE job_id=(SELECT job_id FROM jobs WHERE state=? AND
(lease_expires IS NULL OR lease_expires < ?) ORDER BY job_id LIMIT 1)
AND state=? RETURNING job_id,spec"""
HEARTBEAT_SQL = """UPDATE jobs SET lease_expires=?, updated=? WHERE job_id=?
AND lease_owner=? AND lease_expires>=? AND state IN ('rendering','postprocessing')"""
COMPLETE_SQL = """UPDATE jobs SET state=?, lease_owner=NULL, lease_expires=NULL,
updated=?, error=NULL, render_seconds=COALESCE(?,render_seconds),
post_seconds=COALESCE(?,post_seconds) WHERE job_id=? AND lease_owner=?
AND lease_expires>=? AND state IN ('rendering','postprocessing')"""
FAIL_SQL = """UPDATE jobs SET attempts=attempts+1,
state=CASE WHEN attempts+1>=? THEN 'failed' ELSE ? END,
error=?, lease_owner=NULL, lease_expires=NULL, updated=?
WHERE job_id=? AND lease_owner=? AND lease_expires>=?
AND state IN ('rendering','postprocessing')"""


def connect(path):
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.execute('PRAGMA busy_timeout=30000')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    return db


class Interrupted(BaseException):
    pass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--max-jobs', type=int, default=40)
    parser.add_argument('--lease', type=float, default=900)
    parser.add_argument('--stub', action='store_true')
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args(argv)
    if args.lease <= 0 or args.max_jobs < 1:
        parser.error('lease and max-jobs must be positive')
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    db_path = args.run_dir / 'manifest.sqlite'
    db = connect(db_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    inflight = args.run_dir / 'inflight' / (args.owner + '.json')
    inflight.parent.mkdir(exist_ok=True)
    completed = 0
    def parent_alive():
        return args.parent_pid is None or os.getppid() == args.parent_pid
    try:
        while completed < args.max_jobs and not stop.is_set() and parent_alive():
            if (args.run_dir / 'STOP').exists():
                break
            if (args.run_dir / '.PAUSE_CLAIMS').exists():
                stop.wait(.2)
                continue
            now = time.time()
            row = db.execute(CLAIM_SQL, ('rendering', args.owner, now+args.lease, now,
                                       'pending', now, 'pending')).fetchone()
            if not row:
                break
            job_id, encoded = row
            record = inflight.with_suffix('.tmp')
            record.write_text(json.dumps(dict(job_id=job_id, owner=args.owner)))
            os.replace(record, inflight)
            spec = json.loads(encoded)
            job_dir = args.run_dir / 'jobs' / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / 'spec.json').write_text(json.dumps(spec, sort_keys=True))
            # Owner-specific staging prevents an expired renderer corrupting a new claim.
            temporary = job_dir / ('.tmp_raw-' + args.owner)
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir()
            heartbeat_stop = threading.Event()
            lost = threading.Event()
            def pulse():
                with connect(db_path) as pulse_db:
                    while not heartbeat_stop.wait(min(30., args.lease / 3)):
                        if not parent_alive():
                            stop.set()
                        now = time.time()
                        if not pulse_db.execute(HEARTBEAT_SQL, (now+args.lease, now,
                                                               job_id, args.owner, now)).rowcount:
                            lost.set()
                            return
            thread = threading.Thread(target=pulse, daemon=True)
            thread.start()
            def progress(*_a, **_kw):
                if stop.is_set() or lost.is_set() or not parent_alive():
                    raise Interrupted()
                now = time.time()
                if not db.execute(HEARTBEAT_SQL, (now+args.lease, now,
                                                 job_id, args.owner, now)).rowcount:
                    raise Interrupted()
            started = time.monotonic()
            try:
                if args.stub:
                    if os.environ.get('SCYTHE_STUB_CRASH_JOB') == job_id:
                        os._exit(139)
                    import numpy as np
                    metadata = dict(job_id=job_id, stub=True, blender_version='stub',
                                    camera=dict(profile=spec['isp_variants'][0]['profile'],
                                                resolution=spec['render']['resolution']), keyframes=[])
                    for i, t in enumerate(spec['keyframes']):
                        time.sleep(float(os.environ.get('SCYTHE_STUB_FRAME_SECONDS', '.01')))
                        with (temporary / f'k{i:03d}.exr').open('wb') as f:
                            np.save(f, np.zeros((2, 2, 4), dtype=np.float16))
                        metadata['keyframes'].append(dict(idx=i, t=t, telemetry={},
                            failure_state=dict(active=False, severity=0.), class_pixels={}, events_active=[]))
                        progress()
                    metadata['render_seconds'] = time.monotonic()-started
                    (temporary / 'render_meta.json').write_text(json.dumps(metadata))
                else:
                    from scenegen.sequence import render_sequence
                    render_sequence(spec, temporary, progress=progress)
                progress()
                if not (temporary / 'render_meta.json').is_file() or any(
                        not (temporary / f'k{i:03d}.exr').is_file() for i in range(len(spec['keyframes']))):
                    raise RuntimeError('Renderer returned incomplete raw outputs')
                # Persist raw bytes before recording a successful render.
                for path in temporary.iterdir():
                    if path.is_file():
                        with path.open('rb') as stream:
                            os.fsync(stream.fileno())
                descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                # Fence publication and state transition in one SQLite write transaction.
                db.execute('BEGIN IMMEDIATE')
                try:
                    progress()
                    raw = job_dir / 'raw'
                    if raw.exists():
                        shutil.rmtree(raw)
                    os.replace(temporary, raw)
                    descriptor = os.open(job_dir, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    now = time.time()
                    changed = db.execute(COMPLETE_SQL, ('rendered', now, time.monotonic()-started,
                                                         None, job_id, args.owner, now)).rowcount
                    if not changed:
                        raise Interrupted()
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise
                completed += 1
            except Interrupted:
                stop.set()
            except Exception:
                error = traceback.format_exc()
                (job_dir / 'render_error.log').write_text(error)
                now = time.time()
                db.execute(FAIL_SQL, (3, 'pending', error+'\nlog: '+str(job_dir / 'render_error.log'),
                                     now, job_id, args.owner, now))
                completed += 1
            finally:
                heartbeat_stop.set()
                thread.join(timeout=35)
                shutil.rmtree(temporary, ignore_errors=True)
                if sys.exc_info()[0] is None:
                    inflight.unlink(missing_ok=True)
    finally:
        # Unexpected startup/staging exceptions must remain attributable to this
        # worker; only a normal exit (including handled shutdown) releases cleanly.
        if sys.exc_info()[0] is None:
            db.execute("""UPDATE jobs SET state='pending',lease_owner=NULL,lease_expires=NULL,
                       updated=? WHERE state='rendering' AND lease_owner=?""", (time.time(), args.owner))
            inflight.unlink(missing_ok=True)
        db.close()
    return 0


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
