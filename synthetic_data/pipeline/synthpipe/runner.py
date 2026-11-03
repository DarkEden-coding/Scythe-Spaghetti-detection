"""Resumable render/post supervisor with fenced publication and bounded restarts."""
import concurrent.futures
import fcntl
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

from synthpipe import paths
from synthpipe.manifest import Manifest
from synthpipe.sampler import run_dir
from synthpipe.version import GENERATOR_VERSION, code_hash


class RunLock:
    """Kernel lock survives stale PID files, and is released even after SIGKILL."""
    def __init__(self, directory):
        self.path = directory / 'supervisor.lock'
        self.file = None

    def __enter__(self):
        self.file = self.path.open('a+')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError('Another supervisor holds '+str(self.path)) from None
        self.file.seek(0)
        self.file.truncate()
        self.file.write(json.dumps(dict(pid=os.getpid(), started=time.time())))
        self.file.flush()
        return self

    def __exit__(self, *_):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


def cleanup_temporary(directory, manifest):
    """Never remove staging belonging to a live lease (including orphan workers)."""
    now = time.time()
    for job in (directory / 'jobs').glob('*'):
        row = manifest.db.execute('SELECT lease_owner,lease_expires FROM jobs WHERE job_id=?',
                                  (job.name,)).fetchone()
        if row and row['lease_owner'] and (row['lease_expires'] or 0) >= now:
            continue
        for temporary in job.glob('.tmp_*'):
            if temporary.is_dir():
                shutil.rmtree(temporary)
            else:
                temporary.unlink(missing_ok=True)


def _post_initializer(parent_pid):
    # Idle ProcessPoolExecutor children otherwise retain queue handles after SIGKILL.
    def watch_parent():
        while os.getppid() == parent_pid:
            time.sleep(.2)
        os._exit(0)
    threading.Thread(target=watch_parent, daemon=True).start()


class PostInterrupted(Exception):
    pass


def sync_tree(directory):
    for path in directory.rglob('*'):
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    for path in sorted((p for p in directory.rglob('*') if p.is_dir() and not p.is_symlink()),
                       key=lambda p: len(p.parts), reverse=True):
        sync_directory(path)
    sync_directory(directory)


def sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _post_job(directory, job_id, spec, owner, lease, stub, keep_raw, parent_pid, stop_file):
    directory = Path(directory)
    job = directory / 'jobs' / job_id
    stop = threading.Event()
    lost = threading.Event()
    def pulse():
        with Manifest(directory / 'manifest.sqlite') as manifest:
            while not stop.wait(min(30., lease/3)):
                if os.getppid() != parent_pid or not manifest.heartbeat(job_id, owner, lease):
                    lost.set()
                    return
    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    started = time.monotonic()
    def check_running():
        if lost.is_set() or os.getppid() != parent_pid or Path(stop_file).exists():
            raise PostInterrupted('Postprocessing interrupted or lease lost')
    temporary = job / ('.tmp_post-'+owner)
    try:
        check_running()
        temporary.mkdir(parents=True, exist_ok=True)
        if stub:
            (temporary / 'labels.json').write_text(json.dumps(dict(
                job_id=job_id, spec=spec, stub=True, keyframes=[], variants=[], windows=spec['windows'])))
        else:
            from synthpipe.postprocess import process_job
            # Isolate its writes; publish under our lease fence below.
            def check():
                check_running()
                with Manifest(directory / 'manifest.sqlite') as beat:
                    if not beat.heartbeat(job_id, owner, lease):
                        raise PostInterrupted('Postprocessing lease lost')
            (temporary / 'raw').symlink_to(job / 'raw', target_is_directory=True)
            process_job(temporary, spec, keep_raw=True, heartbeat=check)
            (temporary / 'raw').unlink()
        sync_tree(temporary)
        with Manifest(directory / 'manifest.sqlite') as manifest:
            manifest.db.execute('BEGIN IMMEDIATE')
            try:
                check_running()
                if not manifest.heartbeat(job_id, owner, lease):
                    raise PostInterrupted('Postprocessing lease lost')
                if not stub:
                    for name in ('frames', 'masks'):
                        if (job / name).exists():
                            shutil.rmtree(job / name)
                        os.replace(temporary / name, job / name)
                os.replace(temporary / 'labels.json', job / 'labels.json')
                temporary.rmdir()
                sync_directory(job)
                if not (job / 'labels.json').is_file():
                    raise RuntimeError('Postprocessor did not publish labels.json')
                if not stub and (not (job / 'frames').is_dir() or not (job / 'masks').is_dir()):
                    raise RuntimeError('Postprocessor did not publish frames/masks')
                if not manifest.complete(job_id, owner, 'done', post_seconds=time.monotonic()-started):
                    raise RuntimeError('Postprocessing completion lost its lease')
                manifest.db.commit()
            except BaseException:
                manifest.db.rollback()
                raise
        # Deletion after done is safe: an interrupted deletion only keeps extra raw data.
        retained = int(hashlib.sha256(job_id.encode()).hexdigest()[:16], 16)/2**64 < keep_raw
        if not retained:
            # NORMAL commits need a checkpoint before destructive raw deletion.
            with Manifest(directory / 'manifest.sqlite') as durable:
                checkpoint = durable.db.execute('PRAGMA wal_checkpoint(FULL)').fetchone()
            if checkpoint[0] == 0:
                shutil.rmtree(job / 'raw', ignore_errors=True)
        return True
    except PostInterrupted:
        with Manifest(directory / 'manifest.sqlite') as manifest:
            manifest.release_owner(owner)
        return False
    except Exception:
        error = traceback.format_exc()
        (job / 'post_error.log').write_text(error)
        with Manifest(directory / 'manifest.sqlite') as manifest:
            if os.getppid() == parent_pid:
                manifest.fail(job_id, owner, error+'\nlog: '+str(job / 'post_error.log'), 'rendered')
        return False
    finally:
        stop.set()
        thread.join(timeout=35)
        shutil.rmtree(temporary, ignore_errors=True)


def record_provenance(directory, allow_code_change=False):
    path = directory / 'run.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    digest = code_hash()
    if config.get('code_hash') not in (None, digest) and not allow_code_change:
        raise RuntimeError('Generator code_hash differs from run.json; use --allow-code-change to resume')
    config.update(code_hash=digest, generator_version=GENERATOR_VERSION)
    config.setdefault('code_sessions', []).append(dict(
        code_hash=digest, generator_version=GENERATOR_VERSION, started=time.time()))
    temporary = directory / '.run.json.runner.tmp'
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return config


def vdo_free_bytes(directory):
    """Read physical free space for VDO backing this filesystem, if accessible."""
    def command(*args):
        return subprocess.run(args, capture_output=True, text=True, timeout=5, check=True).stdout
    try:
        source = command('findmnt', '-n', '-o', 'SOURCE', '-T', str(directory)).strip()
        devices = command('lsblk', '-s', '-l', '-n', '-p', '-o', 'NAME', source).splitlines()
        devices = {os.path.realpath(device.strip()) for device in devices + [source]}
        free = []
        for line in command('vdostats').splitlines():
            fields = line.split()
            if len(fields) >= 4 and os.path.realpath(fields[0]) in devices:
                free.append(int(fields[3]) * 1024)  # vdostats reports 1K blocks.
        return min(free) if free else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def recover_render_worker(directory, manifest, owner, code):
    inflight = directory / 'inflight' / (owner + '.json')
    try:
        job_id = json.loads(inflight.read_text()).get('job_id')
    except (OSError, ValueError):
        job_id = None
    if code:
        jobs = manifest.crash_owner(owner, code, job_id)
    else:
        manifest.release_owner(owner)
        jobs = []
    inflight.unlink(missing_ok=True)
    return jobs or ([job_id] if code and job_id else [])


def add_arguments(parser):
    parser.add_argument('--run', required=True)
    parser.add_argument('--render-workers', type=int, default=1)
    parser.add_argument('--post-workers', type=int, default=6)
    parser.add_argument('--max-jobs-per-blender', type=int, default=40)
    parser.add_argument('--stub', action='store_true')
    parser.add_argument('--until', type=int)
    parser.add_argument('--lease', type=float, default=900)
    parser.add_argument('--min-free-gb', type=float, default=40.)
    parser.add_argument('--poll-seconds', type=float, default=1.)
    parser.add_argument('--allow-code-change', action='store_true')
    parser.add_argument('--max-crashes', type=int, default=5)


def run(args):
    directory = run_dir(args.run)
    if not (directory / 'manifest.sqlite').is_file():
        raise ValueError('Run has no manifest; sample jobs first')
    if min(args.render_workers, args.post_workers, args.max_jobs_per_blender, args.max_crashes) < 1:
        raise ValueError('worker counts, max-jobs and max-crashes must be positive')
    if args.lease <= 0 or args.poll_seconds <= 0 or args.min_free_gb < 0:
        raise ValueError('Invalid lease, poll interval or free space threshold')
    if args.until is not None and args.until < 0:
        raise ValueError('until must be nonnegative')
    try:
        with RunLock(directory):
            record_provenance(directory, getattr(args, 'allow_code_change', False))
            return supervise(args, directory)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def supervise(args, directory):
    logs = directory / 'logs'
    logs.mkdir(exist_ok=True)
    logger = logging.getLogger('synthpipe.runner.'+uuid.uuid4().hex)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(logs / 'supervisor.log', maxBytes=5_000_000, backupCount=4)
    logger.addHandler(handler)
    stopped = threading.Event()
    previous = {s: signal.signal(s, lambda *_: stopped.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    token = uuid.uuid4().hex
    children = {}  # slot -> (process, owner, start, output)
    crashes = {i: [] for i in range(args.render_workers)}
    crash_jobs = {i: set() for i in crashes}
    retry_at = dict.fromkeys(crashes, 0.)
    futures = {}
    pause = directory / '.PAUSE_CLAIMS'
    stop_file = logs / ('.stop-'+token)
    config = json.loads((directory / 'run.json').read_text()) if (directory / 'run.json').exists() else {}
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.post_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=_post_initializer, initargs=(os.getpid(),))
    exit_code = 0
    next_disk_check = 0.
    physical_free = None
    manifest = Manifest(directory / 'manifest.sqlite')
    rendered_count = manifest.db.execute(
        'SELECT COUNT(*) FROM jobs WHERE render_seconds IS NOT NULL').fetchone()[0]
    try:
        manifest.reclaim_expired()
        cleanup_temporary(directory, manifest)
        logger.info('Starting supervisor pid=%s token=%s', os.getpid(), token)
        while not stopped.is_set():
            if (directory / 'STOP').exists():
                logger.info('STOP file requested shutdown')
                break
            if manifest.reclaim_expired():
                cleanup_temporary(directory, manifest)
            for future, owner in list(futures.items()):
                if future.done():
                    try:
                        future.result()
                    except Exception:
                        logger.exception('Post worker crashed')
                        manifest.release_owner(owner)
                    del futures[future]
            counts = manifest.counts()
            progress = manifest.db.execute(
                'SELECT COUNT(*) FROM jobs WHERE render_seconds IS NOT NULL').fetchone()[0]
            if progress > rendered_count:
                # Successful renders break a systemic crash streak even when a
                # persistent worker crashes later in the same process session.
                for slot in crashes:
                    crashes[slot].clear()
                    crash_jobs[slot].clear()
            rendered_count = progress
            if args.until is not None and counts['done'] >= args.until:
                break
            if not sum(counts[s] for s in ('pending', 'rendering', 'rendered', 'postprocessing')):
                exit_code = int(counts['failed'] > 0)
                break
            if time.monotonic() >= next_disk_check:
                physical_free = vdo_free_bytes(directory)
                next_disk_check = time.monotonic()+30
            threshold = args.min_free_gb*1_000_000_000
            low_disk = (shutil.disk_usage(directory).free < threshold or
                        (physical_free is not None and physical_free < threshold))
            if low_disk:
                if not pause.exists():
                    pause.touch()
                    logger.warning('Low disk space; pausing new claims')
            else:
                pause.unlink(missing_ok=True)
            for slot in range(args.render_workers):
                if slot in children:
                    process, owner, started, output = children[slot]
                    code = process.poll()
                    if code is None:
                        continue
                    output.close()
                    jobs = recover_render_worker(directory, manifest, owner, code)
                    del children[slot]
                    logger.info('Render worker %s exited %s', owner, code)
                    if code:
                        now = time.monotonic()
                        # Repeated crashes of one poison job consume its retries,
                        # rather than the systemic budget. Startup failures count each time.
                        if not jobs or any(job not in crash_jobs[slot] for job in jobs):
                            crashes[slot].append(now)
                        crash_jobs[slot].update(jobs)
                        retry_at[slot] = now + (.1 if jobs else min(30, 2**len(crashes[slot])))
                    else:
                        crashes[slot].clear()
                        crash_jobs[slot].clear()
                if (low_disk or not counts['pending'] or time.monotonic() < retry_at[slot]
                        or len(crashes[slot]) >= args.max_crashes):
                    continue
                owner = token+'-r'+str(slot)+'-'+uuid.uuid4().hex[:8]
                worker = paths.BLENDER_DIR / 'worker.py'
                command = ([sys.executable, str(worker)] if args.stub else
                           [str(paths.BLENDER_BIN), '-b', '--factory-startup', '-P', str(worker), '--'])
                command += ['--run-dir', str(directory), '--owner', owner, '--max-jobs',
                            str(args.max_jobs_per_blender), '--lease', str(args.lease),
                            '--parent-pid', str(os.getpid())]
                if args.stub:
                    command.append('--stub')
                logfile = logs / f'render-{slot}.log'
                if logfile.exists() and logfile.stat().st_size > 5_000_000:
                    for n in range(3, 0, -1):
                        source = logfile if n == 1 else logs / f'render-{slot}.log.{n-1}'
                        if source.exists():
                            source.replace(logs / f'render-{slot}.log.{n}')
                output = logfile.open('ab', buffering=0)
                try:
                    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except OSError:
                    output.close()
                    logger.exception('Cannot launch render worker')
                    crashes[slot].append(time.monotonic())
                    retry_at[slot] = time.monotonic()+min(30, 2**len(crashes[slot]))
                    continue
                children[slot] = process, owner, time.monotonic(), output
            if not low_disk:
                available = args.post_workers-len(futures)
                if args.until is not None:
                    available = min(available, args.until-counts['done']-len(futures))
                for _ in range(max(0, available)):
                    owner = token+'-p-'+uuid.uuid4().hex
                    claimed = manifest.claim('rendered', 'postprocessing', owner, args.lease)
                    if not claimed:
                        break
                    job_id, spec = claimed
                    future = executor.submit(_post_job, str(directory), job_id, spec, owner,
                                             args.lease, args.stub, config.get('keep_raw', .02), os.getpid(), str(stop_file))
                    futures[future] = owner
            if counts['pending'] and not children and all(
                    len(crashes[i]) >= args.max_crashes for i in crashes):
                logger.error('Render crash budget exhausted; resume after fixing worker errors')
                exit_code = 1
                break
            stopped.wait(min(args.poll_seconds, 30.))
    finally:
        logger.info('Stopping workers')
        stop_file.touch()
        for process, *_ in children.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        deadline = time.monotonic()+45
        for process, owner, _, output in children.values():
            try:
                process.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            output.close()
            recover_render_worker(directory, manifest, owner, process.returncode)
        # Stop at post keyframe callbacks; bound shutdown even if native code hangs.
        processes = list((executor._processes or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        post_deadline = time.monotonic()+45
        for process in processes:
            process.join(timeout=max(.1, post_deadline-time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for owner in futures.values():
            manifest.release_owner(owner)
        cleanup_temporary(directory, manifest)
        pause.unlink(missing_ok=True)
        stop_file.unlink(missing_ok=True)
        manifest.close()
        for sig, prior in previous.items():
            signal.signal(sig, prior)
        handler.close()
        logger.removeHandler(handler)
    return exit_code
