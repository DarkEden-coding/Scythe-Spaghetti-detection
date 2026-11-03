"""Real subprocess tests, including SIGKILL and restart of a 20-job run."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from synthpipe.manifest import Manifest
from synthpipe.runner import RunLock
from test_sampler import fixture_pool
import pytest


def cli_env(root):
    return dict(os.environ, SCYTHE_SYNTH_ROOT=str(root), SCYTHE_STUB_FRAME_SECONDS='.06')


def command(*args):
    # Patch before CLI imports sampler/runner: both bind code_hash directly.
    # Subprocesses cannot inherit pytest's in-process monkeypatches, and agents
    # may legitimately edit the live generator sources between resume sessions.
    script = (
        'from unittest.mock import patch\n'
        'from synthpipe.cli import main\n'
        "with patch('synthpipe.version.code_hash', side_effect="
        "lambda short=False, **kwargs: 'a' * (16 if short else 64)):\n"
        '    raise SystemExit(main())\n'
    )
    return [sys.executable, '-c', script, *args]


def wait_until(predicate, process=None, timeout=40):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if predicate():
            return
        if process is not None and process.poll() is not None:
            raise AssertionError(f'Supervisor exited early: {process.returncode}')
        time.sleep(.05)
    raise AssertionError('Timed out')


def states(directory):
    with Manifest(directory/'manifest.sqlite') as manifest:
        return manifest.counts()


def sample(root, count=20):
    fixture_pool(root)
    result = subprocess.run(command('sample', '--run', 'test', '--count', str(count), '--seed', '12'),
                            env=cli_env(root), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return root/'runs'/'test'


def run_command():
    return command('run', '--run', 'test', '--stub', '--post-workers', '2',
                   '--max-jobs-per-blender', '3', '--lease', '1.5', '--poll-seconds', '.05',
                   '--min-free-gb', '0')


def test_kill_restart_20_jobs(tmp_path):
    directory = sample(tmp_path)
    supervisor = subprocess.Popen(run_command(), env=cli_env(tmp_path))
    try:
        wait_until(lambda: states(directory)['done'] >= 2 and states(directory)['rendering'] > 0,
                   supervisor)
        child_ids = Path(f'/proc/{supervisor.pid}/task/{supervisor.pid}/children').read_text().split()
        supervisor.kill()
        supervisor.wait(timeout=10)
        def children_gone():
            for pid in child_ids:
                try:
                    if Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z':
                        return False
                except FileNotFoundError:
                    pass
            return True
        wait_until(children_gone, timeout=10)
        # A partial directory models a power-loss artifact as well.
        partial = directory/'jobs'/'orphan'/'.tmp_raw'
        partial.mkdir(parents=True)
        (partial/'partial.exr').write_bytes(b'partial')
        resumed = subprocess.run(run_command(), env=cli_env(tmp_path), capture_output=True,
                                 text=True, timeout=80)
        assert resumed.returncode == 0, resumed.stderr
        assert states(directory)['done'] == 20
        assert states(directory)['failed'] == 0
        assert not list(directory.glob('jobs/*/.tmp_*'))
        for job in directory.glob('jobs/s*'):
            assert (job/'labels.json').is_file()
        with Manifest(directory/'manifest.sqlite') as manifest:
            assert manifest.db.execute('SELECT COUNT(DISTINCT job_id) FROM jobs').fetchone()[0] == 20
            # The orphan may release cleanly after detecting its dead parent,
            # or lose its rendering lease first. Only the latter consumes a retry.
            attempts, crash_attempts = manifest.db.execute(
                'SELECT SUM(attempts),SUM(crash_attempts) FROM jobs').fetchone()
            assert attempts == crash_attempts
            assert 0 <= attempts <= 1
            assert manifest.db.execute('SELECT COUNT(*) FROM jobs WHERE lease_owner IS NOT NULL').fetchone()[0] == 0
            before = list(manifest.db.execute('SELECT job_id,updated FROM jobs ORDER BY job_id'))
        assert subprocess.run(run_command(), env=cli_env(tmp_path), timeout=20).returncode == 0
        with Manifest(directory/'manifest.sqlite') as manifest:
            assert [tuple(r) for r in before] == [tuple(r) for r in manifest.db.execute(
                'SELECT job_id,updated FROM jobs ORDER BY job_id')]
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_stop_duplicate_and_resume(tmp_path):
    directory = sample(tmp_path, 8)
    supervisor = subprocess.Popen(run_command(), env=cli_env(tmp_path))
    try:
        wait_until(lambda: states(directory)['rendering'] > 0, supervisor)
        duplicate = subprocess.run(run_command(), env=cli_env(tmp_path), capture_output=True,
                                   text=True, timeout=10)
        assert duplicate.returncode == 2
        assert 'Another supervisor' in duplicate.stderr
        (directory/'STOP').touch()
        assert supervisor.wait(timeout=30) == 0
        assert states(directory)['rendering'] == states(directory)['postprocessing'] == 0
        with Manifest(directory/'manifest.sqlite') as manifest:
            assert manifest.db.execute('SELECT SUM(attempts) FROM jobs').fetchone()[0] == 0
        assert not list((directory/'inflight').glob('*.json'))
        assert not list(directory.glob('jobs/*/.tmp_*'))
        (directory/'STOP').unlink()
        result = subprocess.run(run_command(), env=cli_env(tmp_path), timeout=50)
        assert result.returncode == 0
        assert states(directory)['done'] == 8
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=30)


def test_lock_and_disk_pause(tmp_path):
    directory = sample(tmp_path, 2)
    with RunLock(directory):
        with pytest.raises(RuntimeError, match='Another supervisor'):
            with RunLock(directory):
                pass
    cmd = run_command()+['--min-free-gb', '100000000']
    supervisor = subprocess.Popen(cmd, env=cli_env(tmp_path))
    try:
        wait_until(lambda: (directory/'.PAUSE_CLAIMS').exists(), supervisor)
        assert states(directory)['pending'] == 2
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=20) == 0
        assert states(directory)['pending'] == 2
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_worker_crash_restart_and_until(tmp_path):
    directory = sample(tmp_path, 6)
    supervisor = subprocess.Popen(run_command()+['--until', '3'], env=cli_env(tmp_path))
    try:
        wait_until(lambda: states(directory)['rendering'] > 0, supervisor)
        child_ids = Path(f'/proc/{supervisor.pid}/task/{supervisor.pid}/children').read_text().split()
        killed = False
        for child_id in child_ids:
            try:
                cmdline = Path(f'/proc/{child_id}/cmdline').read_bytes()
                if b'worker.py' in cmdline:
                    os.kill(int(child_id), signal.SIGKILL)
                    killed = True
                    break
            except FileNotFoundError:
                continue
        assert killed
        assert supervisor.wait(timeout=50) == 0
        assert states(directory)['done'] == 3
        result = subprocess.run(run_command(), env=cli_env(tmp_path), timeout=50)
        assert result.returncode == 0
        assert states(directory)['done'] == 6
        status = subprocess.run(command('status', '--run', 'test'), env=cli_env(tmp_path),
                                capture_output=True, text=True, timeout=20)
        report = json.loads(status.stdout)
        assert report['counts']['done'] == 6
        assert report['avg_render_seconds'] > 0
        assert report['avg_post_seconds'] > 0
        assert report['jobs_per_hour'] == 6
        assert report['crash_attempts'] == 1
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=30)


def test_postprocessor_staging_and_interrupt(tmp_path, monkeypatch):
    from synthpipe.runner import _post_job
    import synthpipe.postprocess as postprocess
    directory = sample(tmp_path, 1)
    stop = directory/'stop-post'
    with Manifest(directory/'manifest.sqlite') as manifest:
        job_id, spec = manifest.claim('pending', 'rendering', 'render', 60)
        manifest.complete(job_id, 'render', 'rendered')
        manifest.claim('rendered', 'postprocessing', 'post', 60)
    job = directory/'jobs'/job_id
    (job/'raw').mkdir(parents=True)
    (job/'raw'/'source').write_bytes(b'render')
    def processor(stage, spec, *, keep_raw, heartbeat):
        assert stage.name.startswith('.tmp_post-')
        assert (stage/'raw'/'source').read_bytes() == b'render'
        assert keep_raw
        heartbeat()
        for folder in ('frames', 'masks'):
            (stage/folder).mkdir()
            (stage/folder/'output').write_bytes(b'output')
        (stage/'labels.json').write_text(json.dumps(dict(job_id=spec['job_id'])))
    monkeypatch.setattr(postprocess, 'process_job', processor)
    assert _post_job(directory, job_id, spec, 'post', 60, False, 1., os.getppid(), str(stop))
    assert states(directory)['done'] == 1
    assert (job/'frames'/'output').is_file()
    assert (job/'raw'/'source').is_file()
    assert not list(job.glob('.tmp_*'))
    # Interruption releases a claimed job without consuming a retry.
    with Manifest(directory/'manifest.sqlite') as manifest:
        manifest.db.execute("UPDATE jobs SET state='rendered'")
        manifest.claim('rendered', 'postprocessing', 'post2', 60)
    stop.touch()
    assert not _post_job(directory, job_id, spec, 'post2', 60, False, 1., os.getppid(), str(stop))
    assert states(directory)['rendered'] == 1
    with Manifest(directory/'manifest.sqlite') as manifest:
        assert manifest.db.execute('SELECT attempts FROM jobs').fetchone()[0] == 0


def test_render_crash_budget(tmp_path):
    directory = sample(tmp_path, 1)
    env = dict(cli_env(tmp_path), SCYTHE_BLENDER='/bin/false')
    cmd = command('run', '--run', 'test', '--max-crashes', '2', '--poll-seconds', '.05',
                  '--min-free-gb', '0', '--post-workers', '1')
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert states(directory)['pending'] == 1
    assert 'crash budget exhausted' in (directory/'logs'/'supervisor.log').read_text()


def test_stale_post_cannot_publish(tmp_path, monkeypatch):
    from synthpipe.runner import _post_job
    import synthpipe.postprocess as postprocess
    directory = sample(tmp_path, 1)
    with Manifest(directory/'manifest.sqlite') as manifest:
        job_id, spec = manifest.claim('pending', 'rendering', 'render', 60)
        manifest.complete(job_id, 'render', 'rendered')
        manifest.claim('rendered', 'postprocessing', 'expired', 60)
    job = directory/'jobs'/job_id
    (job/'raw').mkdir(parents=True)
    (job/'labels.json').write_text('original')
    def processor(stage, spec, **kwargs):
        for folder in ('frames', 'masks'):
            (stage/folder).mkdir()
        (stage/'labels.json').write_text('stale output')
        with Manifest(directory/'manifest.sqlite') as manifest:
            manifest.db.execute('UPDATE jobs SET lease_expires=0')
            manifest.reclaim_expired()
            assert manifest.claim('rendered', 'postprocessing', 'current', 60)
    monkeypatch.setattr(postprocess, 'process_job', processor)
    assert not _post_job(directory, job_id, spec, 'expired', 60, False, 1.,
                         os.getppid(), str(directory/'stop'))
    assert (job/'labels.json').read_text() == 'original'
    assert states(directory)['postprocessing'] == 1
    with Manifest(directory/'manifest.sqlite') as manifest:
        row = manifest.db.execute('SELECT lease_owner,attempts FROM jobs').fetchone()
        assert tuple(row) == ('current', 0)
    assert not list(job.glob('.tmp_*'))


@pytest.mark.parametrize('stop_signal', [signal.SIGINT, signal.SIGTERM])
def test_signal_shutdown_releases_without_attempt(tmp_path, stop_signal):
    directory = sample(tmp_path, 2)
    supervisor = subprocess.Popen(run_command(), env=cli_env(tmp_path))
    try:
        wait_until(lambda: states(directory)['rendering'] > 0, supervisor)
        supervisor.send_signal(stop_signal)
        assert supervisor.wait(timeout=30) == 0
        assert states(directory)['rendering'] == states(directory)['postprocessing'] == 0
        with Manifest(directory/'manifest.sqlite') as manifest:
            assert manifest.db.execute('SELECT SUM(attempts) FROM jobs').fetchone()[0] == 0
        assert not list(directory.glob('jobs/*/.tmp_*'))
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_sample_dryrun_idempotence_and_config(tmp_path):
    fixture_pool(tmp_path)
    args = command('sample', '--run', 'test', '--count', '20', '--seed', '12')
    dry = subprocess.run(args+['--dry-run'], env=cli_env(tmp_path), capture_output=True, text=True, timeout=30)
    assert dry.returncode == 0, dry.stderr
    assert not (tmp_path/'runs'/'test').exists()
    for _ in range(2):
        result = subprocess.run(args, env=cli_env(tmp_path), capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    directory = tmp_path/'runs'/'test'
    assert states(directory)['pending'] == 20
    assert len((directory/'specs.jsonl').read_text().splitlines()) == 20
    config = json.loads((directory/'run.json').read_text())
    assert len(config['git_hash']) == 40
    assert isinstance(config['git_dirty'], bool)
    assert config['render']['resolution_by_profile']['pi_noir'] == [640,480]
    wrong = subprocess.run(args+['--seed', '13'], env=cli_env(tmp_path), capture_output=True, text=True, timeout=30)
    assert wrong.returncode != 0
    assert states(directory)['pending'] == 20


def test_poison_job_quarantined_without_exhausting_budget(tmp_path):
    directory = sample(tmp_path, 7)
    with Manifest(directory/'manifest.sqlite') as manifest:
        poison = manifest.db.execute('SELECT job_id FROM jobs ORDER BY job_id LIMIT 1').fetchone()[0]
    env = dict(cli_env(tmp_path), SCYTHE_STUB_CRASH_JOB=poison)
    result = subprocess.run(run_command()+['--max-crashes', '2'], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, result.stderr  # A failed job makes the final result nonzero.
    assert states(directory)['done'] == 6
    assert states(directory)['failed'] == 1
    with Manifest(directory/'manifest.sqlite') as manifest:
        row = manifest.db.execute('SELECT attempts,crash_attempts,error FROM jobs WHERE job_id=?',
                                  (poison,)).fetchone()
        assert tuple(row) == (3, 3, 'worker crashed (exit code 139)')
    assert not list((directory/'inflight').glob('*.json'))
    assert 'crash budget exhausted' not in (directory/'logs'/'supervisor.log').read_text()
    report = subprocess.run(command('status', '--run', 'test'), env=env,
                            capture_output=True, text=True, timeout=20)
    status = json.loads(report.stdout)
    assert status['crash_attempts'] == 3
    assert status['jobs_done_last_15m'] == status['jobs_done_last_60m'] == 6
    assert status['jobs_per_hour_15m'] == 24
    assert status['jobs_per_hour_60m'] == 6


def test_code_hash_mismatch_refusal_and_override(tmp_path):
    directory = sample(tmp_path, 1)
    result = subprocess.run(run_command(), env=cli_env(tmp_path), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    path = directory/'run.json'
    config = json.loads(path.read_text())
    assert config['code_hash'] == 'a'*64
    assert config['generator_version']
    config['code_hash'] = 'old-generator'
    path.write_text(json.dumps(config))
    refused = subprocess.run(run_command(), env=cli_env(tmp_path), capture_output=True,
                             text=True, timeout=20)
    assert refused.returncode == 2
    assert '--allow-code-change' in refused.stderr
    assert json.loads(path.read_text())['code_hash'] == 'old-generator'
    allowed = subprocess.run(run_command()+['--allow-code-change'], env=cli_env(tmp_path),
                             capture_output=True, text=True, timeout=20)
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(path.read_text())['code_hash'] == 'a'*64


def test_vdo_physical_free_space_and_unavailable(tmp_path, monkeypatch):
    from synthpipe.runner import vdo_free_bytes
    def run(args, **kwargs):
        outputs = {'findmnt': '/dev/mapper/run-volume\n',
                   'lsblk': '/dev/mapper/run-volume\n/dev/mapper/vdo-backing\n',
                   'vdostats': 'Device 1K-blocks Used Available Use% Space saving%\n'
                               '/dev/mapper/unrelated 1000 990 10 99% 0%\n'
                               '/dev/mapper/vdo-backing 1000 800 200 80% 0%\n'}
        return subprocess.CompletedProcess(args, 0, outputs[args[0]], '')
    monkeypatch.setattr(subprocess, 'run', run)
    assert vdo_free_bytes(tmp_path) == 200*1024
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'vdostats')
    monkeypatch.setattr(subprocess, 'run', unavailable)
    assert vdo_free_bytes(tmp_path) is None


def test_status_rolling_windows(tmp_path, capsys, monkeypatch):
    from synthpipe import runner_status
    from argparse import Namespace
    directory = sample(tmp_path, 4)
    now = time.time()
    with Manifest(directory/'manifest.sqlite') as manifest:
        ids = [row[0] for row in manifest.db.execute('SELECT job_id FROM jobs ORDER BY job_id')]
        for job, age in zip(ids, (60, 1200, 7200)):
            manifest.db.execute("UPDATE jobs SET state='done',updated=? WHERE job_id=?", (now-age, job))
    monkeypatch.setattr(runner_status, 'run_dir', lambda _: directory)
    runner_status.run(Namespace(run='test'))
    report = json.loads(capsys.readouterr().out)
    assert report['jobs_done_last_15m'] == 1
    assert report['jobs_done_last_60m'] == 2
    assert report['jobs_per_hour_15m'] == 4
    assert report['jobs_per_hour_60m'] == 2
    assert report['eta_hours'] == .25
