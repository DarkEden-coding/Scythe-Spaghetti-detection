import importlib.util
import json
import multiprocessing
from pathlib import Path
import time

from synthpipe.manifest import Manifest


def claim_all(path, queue):
    claimed = []
    with Manifest(path) as manifest:
        while row := manifest.claim('pending', 'rendering', str(multiprocessing.current_process().pid), 60):
            claimed.append(row[0])
    queue.put(claimed)


def test_multiprocess_claims_never_duplicate(tmp_path):
    path = tmp_path / 'manifest.sqlite'
    with Manifest(path) as manifest:
        manifest.add_jobs([dict(job_id=str(i)) for i in range(120)])
        manifest.add_jobs([dict(job_id='0', changed=True)])
        assert json.loads(manifest.db.execute("SELECT spec FROM jobs WHERE job_id='0'").fetchone()[0]) == {'job_id': '0'}
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    workers = [context.Process(target=claim_all, args=(path, queue)) for _ in range(6)]
    for worker in workers:
        worker.start()
    claims = [job for _ in workers for job in queue.get(timeout=30)]
    for worker in workers:
        worker.join(30)
        assert worker.exitcode == 0
    assert len(claims) == len(set(claims)) == 120


def test_expired_stale_worker_and_release(tmp_path):
    with Manifest(tmp_path/'manifest.sqlite') as manifest:
        manifest.add_jobs([dict(job_id='a')])
        manifest.claim('pending', 'rendering', 'old', .02)
        time.sleep(.03)
        assert not manifest.heartbeat('a', 'old', 60)
        assert not manifest.complete('a', 'old', 'rendered')
        assert manifest.reclaim_expired() == 1
        manifest.claim('pending', 'rendering', 'new', 60)
        assert not manifest.fail('a', 'old', 'stale', 'pending')
        assert manifest.heartbeat('a', 'new', 60)
        assert manifest.complete('a', 'new', 'rendered', render_seconds=2)
        manifest.claim('rendered', 'postprocessing', 'post', 60)
        assert manifest.release_owner('post') == 1
        assert manifest.counts()['rendered'] == 1
        assert manifest.db.execute('SELECT attempts FROM jobs').fetchone()[0] == 1


def test_retry_failed_and_metrics(tmp_path):
    with Manifest(tmp_path/'manifest.sqlite') as manifest:
        manifest.add_jobs([dict(job_id='a')])
        for attempt in range(3):
            assert manifest.claim('pending', 'rendering', 'worker', 60)
            assert manifest.fail('a', 'worker', f'error {attempt}', 'pending')
        assert manifest.counts()['failed'] == 1
        assert manifest.claim('pending', 'rendering', 'worker', 60) is None
        assert manifest.db.execute('SELECT attempts,error FROM jobs').fetchone()[:] == (3, 'error 2')


def test_worker_sql_matches_manifest():
    from synthpipe import paths
    from synthpipe import manifest
    spec = importlib.util.spec_from_file_location('blender_worker', paths.BLENDER_DIR/'worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    for name in ('CLAIM_SQL', 'HEARTBEAT_SQL', 'COMPLETE_SQL', 'FAIL_SQL'):
        assert getattr(worker, name) == getattr(manifest, name)


def test_expired_render_quarantines_but_clean_release_does_not(tmp_path):
    with Manifest(tmp_path/'manifest.sqlite') as manifest:
        manifest.add_jobs([dict(job_id='a')])
        for _ in range(2):
            manifest.claim('pending', 'rendering', 'clean', 60)
            manifest.release_owner('clean')
        assert manifest.db.execute('SELECT attempts FROM jobs').fetchone()[0] == 0
        for attempt in range(1, 4):
            manifest.claim('pending', 'rendering', 'lost', 60)
            manifest.db.execute('UPDATE jobs SET lease_expires=0')
            assert manifest.reclaim_expired() == 1
            assert manifest.reclaim_expired() == 0
            assert manifest.crash_owner('lost', -9, 'a') == []
            row = manifest.db.execute('SELECT attempts,crash_attempts,state,error FROM jobs').fetchone()
            assert tuple(row[:2]) == (attempt, attempt)
            assert row['state'] == ('failed' if attempt == 3 else 'pending')
            assert 'lease expired' in row['error']


def test_crash_owner_fences_all_claims_and_stale_inflight(tmp_path):
    with Manifest(tmp_path/'manifest.sqlite') as manifest:
        manifest.add_jobs([dict(job_id=x) for x in ('a', 'b', 'c')])
        manifest.claim('pending', 'rendering', 'crash', 60)
        manifest.claim('pending', 'rendering', 'crash', 60)
        manifest.claim('pending', 'rendering', 'other', 60)
        assert set(manifest.crash_owner('crash', 139, 'c')) == {'a', 'b'}
        assert manifest.crash_owner('crash', 139, 'a') == []
        rows = list(manifest.db.execute('SELECT job_id,attempts,error FROM jobs ORDER BY job_id'))
        assert [row['attempts'] for row in rows] == [1, 1, 0]
        assert rows[0]['error'] == 'worker crashed (exit code 139)'
