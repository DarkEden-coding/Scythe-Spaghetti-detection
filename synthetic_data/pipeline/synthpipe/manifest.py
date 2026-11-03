"""Durable leased job queue. All mutations fence stale workers by owner and lease."""
import json
import sqlite3
import time
from pathlib import Path

STATES = ('pending', 'rendering', 'rendered', 'postprocessing', 'done', 'failed')
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


class Manifest:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA busy_timeout=30000')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, lease_owner TEXT, lease_expires REAL,
            error TEXT, created REAL NOT NULL, updated REAL NOT NULL,
            render_seconds REAL, post_seconds REAL)''')
        if 'crash_attempts' not in {row[1] for row in self.db.execute('PRAGMA table_info(jobs)')}:
            self.db.execute('ALTER TABLE jobs ADD COLUMN crash_attempts INTEGER NOT NULL DEFAULT 0')
        self.db.execute('CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state,lease_expires)')

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def add_jobs(self, specs):
        now = time.time()
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                self.db.executemany('''INSERT OR IGNORE INTO jobs
                    (job_id,spec,state,created,updated) VALUES (?,?,'pending',?,?)''',
                    ((s['job_id'], json.dumps(s, sort_keys=True), now, now) for s in specs))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def claim(self, from_state, to_state, owner, lease_s):
        if (from_state, to_state) not in {('pending', 'rendering'), ('rendered', 'postprocessing')}:
            raise ValueError('Invalid claim transition')
        if not owner or lease_s <= 0:
            raise ValueError('Owner and positive lease are required')
        now = time.time()
        row = self.db.execute(CLAIM_SQL, (to_state, owner, now+lease_s, now,
                                         from_state, now, from_state)).fetchone()
        return (row['job_id'], json.loads(row['spec'])) if row else None

    def heartbeat(self, job_id, owner, lease_s):
        now = time.time()
        return bool(self.db.execute(HEARTBEAT_SQL,
                    (now+lease_s, now, job_id, owner, now)).rowcount)

    def complete(self, job_id, owner, to_state, **metrics):
        expected = {'rendered': 'rendering', 'done': 'postprocessing'}
        if to_state not in expected:
            raise ValueError('Invalid completion transition')
        now = time.time()
        return bool(self.db.execute(COMPLETE_SQL + ' AND state=?',
                    (to_state, now, metrics.get('render_seconds'), metrics.get('post_seconds'),
                     job_id, owner, now, expected[to_state])).rowcount)

    def fail(self, job_id, owner, error, retry_state, max_attempts=3):
        if retry_state not in ('pending', 'rendered'):
            raise ValueError('Invalid retry state')
        now = time.time()
        return bool(self.db.execute(FAIL_SQL + ' AND state=?',
                    (max_attempts, retry_state, str(error), now, job_id, owner, now,
                     'rendering' if retry_state == 'pending' else 'postprocessing')).rowcount)

    def reclaim_expired(self):
        now = time.time()
        return self.db.execute("""UPDATE jobs SET
            attempts=attempts+CASE WHEN state='rendering' THEN 1 ELSE 0 END,
            crash_attempts=crash_attempts+CASE WHEN state='rendering' THEN 1 ELSE 0 END,
            error=CASE WHEN state='rendering' THEN 'worker lease expired while rendering' ELSE error END,
            state=CASE WHEN state='postprocessing' THEN 'rendered'
                       WHEN attempts+1>=3 THEN 'failed' ELSE 'pending' END,
            lease_owner=NULL,lease_expires=NULL,updated=?
            WHERE state IN ('rendering','postprocessing') AND lease_expires < ?""",
            (now, now)).rowcount

    def crash_owner(self, owner, exit_code, inflight_job=None):
        """Charge only claims still owned by this worker, including expired leases.

        The owner fence also prevents a stale in-flight file charging a new claim
        or double charging a job already recovered by lease expiry.
        """
        rows = self.db.execute("""UPDATE jobs SET attempts=attempts+1,
            crash_attempts=crash_attempts+1,
            state=CASE WHEN attempts+1>=3 THEN 'failed' ELSE 'pending' END,
            error=?,lease_owner=NULL,lease_expires=NULL,updated=?
            WHERE state='rendering' AND lease_owner=? RETURNING job_id""",
            (f'worker crashed (exit code {exit_code})', time.time(), owner)).fetchall()
        return [row['job_id'] for row in rows]

    def release_owner(self, owner):
        return self.db.execute('''UPDATE jobs SET state=CASE state
            WHEN 'rendering' THEN 'pending' ELSE 'rendered' END,
            lease_owner=NULL,lease_expires=NULL,updated=? WHERE lease_owner=?
            AND state IN ('rendering','postprocessing')''', (time.time(), owner)).rowcount

    def counts(self):
        counts = dict.fromkeys(STATES, 0)
        counts.update({r['state']: r['n'] for r in
                       self.db.execute('SELECT state,COUNT(*) n FROM jobs GROUP BY state')})
        return counts
