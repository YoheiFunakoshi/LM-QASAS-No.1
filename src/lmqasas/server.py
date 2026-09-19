"""Loopback-only, single-job HTTP interface for local research analysis.

This server is intentionally not a network service. Uploaded inputs are copied
to private, new directories; source CSVs and previous runs are never modified.
"""
from __future__ import annotations

import base64
import binascii
import csv
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[2]
PHASES = ('Pre', 'Peak', 'Post')
CSV_COLUMNS = {'Vseg', 'Jseg', 'CDR3', 'AAlength', 'NTlength', 'Type', 'Cseg', 'Counts', 'Frequency(%)'}
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 20 * 1024 * 1024
ID = re.compile(r'[A-Za-z0-9_-]{1,100}\Z')
STAGES = {'queued', 'loading', 'validated', 'embedding', 'kmeans', 'analysis_completed',
          'projection', 'rendering', 'reselection', 'completed', 'failed', 'interrupted'}
JOB_FIELDS = ('id', 'type', 'status', 'stage', 'progress', 'created_at', 'updated_at', 'result', 'error')
DOWNLOAD_NAMES = {'run_metadata.json', 'input_audit.json', 'cluster_scores.csv',
                  'candidates.csv', 'candidate_provenance.json', 'selection_summary.json',
                  'selection_source.json', 'projection_metadata.json', 'view_metadata.json', 'comparison.png'}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_atomic_json(path: Path, value: dict) -> None:
    """Atomic replacement only for an application's own mutable job record."""
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def process_alive(pid) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


def protect_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    (path / '.gitignore').write_text('# Private local research data.\n*\n', encoding='utf-8')


def confined(base: Path, relative: str, *, require_exists: bool = True) -> Path:
    """Resolve symlinks/junctions before enforcing a strict path boundary."""
    if not isinstance(relative, str) or '\\' in relative or ':' in relative or relative.startswith('/'):
        raise ApiError(400, '保存先の指定が不正です。')
    if any(part in ('', '.', '..') for part in relative.split('/')):
        raise ApiError(400, '保存先の指定が不正です。')
    try:
        base = base.resolve(strict=require_exists)
        result = (base / relative).resolve(strict=require_exists)
    except (OSError, RuntimeError):
        raise ApiError(404, '指定した結果が見つかりません。') from None
    if result == base or base not in result.parents:
        raise ApiError(403, 'この場所のファイルにはアクセスできません。')
    return result


def checked_id(value, prefix: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value) or not value.startswith(prefix):
        raise ApiError(400, '結果の識別名が不正です。')
    return value


def positive_int(value, name: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ApiError(400, f'{name}の値を確認してください。')
    return value


def analysis_parameters(data: dict) -> dict:
    subject = data.get('subject')
    if (not isinstance(subject, str) or not subject.strip() or len(subject) > 100
            or any(ord(char) < 32 for char in subject)):
        raise ApiError(400, '解析用の識別名を1〜100文字で入力してください。')
    defaults = {'top_n': 1000, 'n_clusters': 500, 'n_init': 10, 'batch_size': 32, 'threads': 4}
    limits = {'n_init': 1000, 'batch_size': 1024, 'threads': 256}
    result = {'subject': subject}
    for name, default in defaults.items():
        result[name] = positive_int(data.get(name, default), name, limits.get(name))
    epsilon = data.get('epsilon', 1.0)
    if type(epsilon) not in (int, float):
        raise ApiError(400, 'epsilonは有限の正の数にしてください。')
    try:
        epsilon = float(epsilon)
    except OverflowError:
        raise ApiError(400, 'epsilonは有限の正の数にしてください。') from None
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ApiError(400, 'epsilonは有限の正の数にしてください。')
    seed = data.get('seed', 20260919)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ApiError(400, 'seedは0以上、2の32乗未満の整数にしてください。')
    return {**result, 'epsilon': epsilon, 'seed': seed}


def decode_uploads(files) -> dict:
    if not isinstance(files, dict) or set(files) != set(PHASES):
        raise ApiError(400, 'Pre・Peak・Postの3つのCSVを指定してください。')
    decoded = {}
    for phase in PHASES:
        item = files[phase]
        if not isinstance(item, dict):
            raise ApiError(400, 'CSVの指定形式が不正です。')
        name, content = item.get('name'), item.get('base64')
        if (not isinstance(name, str) or not name.lower().endswith('.csv') or len(name) > 255
                or any(c in name for c in '/\\\x00') or any(ord(c) < 32 for c in name)):
            raise ApiError(400, 'CSVファイルを指定してください。')
        if not isinstance(content, str) or len(content) > 4 * ((MAX_FILE_BYTES + 2) // 3):
            raise ApiError(413, 'CSVは1ファイル20 MiB以下にしてください。')
        try:
            raw = base64.b64decode(content, validate=True)
        except (ValueError, binascii.Error):
            raise ApiError(400, 'CSVの転送形式が不正です。') from None
        if not raw or len(raw) > MAX_FILE_BYTES:
            raise ApiError(413, 'CSVは空でない、20 MiB以下のファイルにしてください。')
        try:
            text = raw.decode('utf-8-sig')
            header = next(csv.reader(io.StringIO(text, newline=''), strict=True), [])
        except (UnicodeError, csv.Error):
            raise ApiError(400, 'CSVの文字コードはUTF-8にしてください。') from None
        if len(header) != len(CSV_COLUMNS) or set(header) != CSV_COLUMNS:
            raise ApiError(400, 'CSVの列名が入力規則と一致しません。日本語解説書を確認してください。')
        decoded[phase] = {'name': name, 'bytes': raw}
    return decoded


class Application:
    def __init__(self, root: Path = ROOT, *, spawn=None):
        self.root = root.resolve(strict=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.process = None
        self.job_path = None
        self.spawn = spawn or subprocess.Popen
        self.jobs = self.root / 'local_records/ui_jobs'
        self.inputs = self.root / 'local_records/ui_inputs'
        self.outputs = self.root / 'outputs'
        # Do not follow relocated private directories outside the project.
        for path in (self.jobs, self.inputs, self.outputs):
            if self.root not in path.resolve().parents:
                raise ValueError('Private directories must remain inside the project.')

    def run_path(self, identifier: str) -> Path:
        if self.root not in self.outputs.resolve().parents:
            raise ApiError(403, '結果の保存先がプロジェクト外へ移動しています。')
        return confined(self.outputs, checked_id(identifier, 'run_'))

    def child_path(self, run: Path, identifier: str, prefix: str) -> Path:
        return confined(run, checked_id(identifier, prefix))

    def job(self) -> dict | None:
        with self.lock:
            if self.job_path is None:
                if self.jobs.exists():
                    old = sorted(self.jobs.glob('job_*/job.json'))
                    if old:
                        try:
                            candidate = old[-1].resolve(strict=True)
                            if self.jobs.resolve() in candidate.parents:
                                self.job_path = candidate
                        except OSError:
                            pass
                if self.job_path is None:
                    return None
            try:
                job = read_json(self.job_path)
            except (OSError, ValueError):
                return None
            if job.get('status') in ('queued', 'running'):
                if self.process is not None:
                    exit_code = self.process.poll()
                else:
                    try:
                        alive = process_alive(read_json(self.job_path.parent / 'process.json').get('pid'))
                    except (OSError, ValueError):
                        alive = False
                    exit_code = None if alive else -1
                if exit_code is not None:
                    # The worker may have committed completion between our first
                    # read and poll. Never overwrite that terminal result.
                    latest = read_json(self.job_path)
                    if latest.get('status') not in ('queued', 'running'):
                        return {key: latest.get(key) for key in JOB_FIELDS}
                    job.update(status='failed', stage='interrupted', progress=None,
                               error='処理が中断されました。完了した結果は保存履歴から確認できます。', updated_at=utc_now())
                    # A restarted server reports old unfinished work but preserves its record.
                    if self.process is not None:
                        write_atomic_json(self.job_path, job)
            return {key: job.get(key) for key in JOB_FIELDS}

    def list_runs(self) -> list[dict]:
        result = []
        if not self.outputs.exists():
            return result
        for directory in sorted(self.outputs.iterdir(), reverse=True):
            if not directory.name.startswith('run_'):
                continue
            try:
                run = self.run_path(directory.name)
                metadata = read_json(confined(run, 'run_metadata.json'))
                result.append({'id': directory.name, 'created_at': metadata.get('created_at_utc'),
                               'subject': metadata.get('subject'), 'status': metadata.get('status'),
                               'selection': metadata.get('selection')})
            except (ApiError, OSError, ValueError):
                continue
        return result

    def run_detail(self, identifier: str, selection_id: str | None = None) -> dict:
        run = self.run_path(identifier)
        metadata = read_json(confined(run, 'run_metadata.json'))
        selections, projections, views = [], [], []
        for directory in sorted(run.iterdir()):
            try:
                if directory.name.startswith('selection_'):
                    checked = self.child_path(run, directory.name, 'selection_')
                    selections.append({'id': directory.name, 'summary': read_json(confined(checked, 'selection_summary.json'))})
                elif directory.name.startswith('projection_'):
                    checked = self.child_path(run, directory.name, 'projection_')
                    projection_metadata = read_json(confined(checked, 'projection_metadata.json'))
                    if projection_metadata.get('status') != 'completed':
                        continue
                    projections.append({'id': directory.name, 'metadata': projection_metadata})
                    for image in checked.rglob('comparison.png'):
                        try:
                            safe_image = confined(run, image.relative_to(run).as_posix())
                            meta = read_json(confined(safe_image.parent, 'view_metadata.json'))
                            if meta.get('status') == 'completed':
                                views.append({'path': safe_image.relative_to(run).as_posix(), 'metadata': meta})
                        except (ApiError, OSError, ValueError):
                            # One interrupted view must not hide completed views.
                            continue
            except (ApiError, OSError, ValueError):
                continue
        if selection_id is None:
            selection_id = 'selection_initial'
        selection = self.child_path(run, selection_id, 'selection_')
        summary = read_json(confined(selection, 'selection_summary.json'))
        with confined(selection, 'candidates.csv').open(encoding='utf-8-sig', newline='') as handle:
            candidates = [{'rank': int(row['rank']), 'cdr3': row['cdr3'], 'score': float(row['score']),
                           'clone_count': int(row['clone_count']), 'cluster_ids': row['cluster_ids']}
                          for row in csv.DictReader(handle)]
        audit = read_json(confined(run, 'input_audit.json'))
        allowed_sample_fields = {'total_rows', 'accepted_rows', 'excluded_rows', 'clone_count', 'merged_rows',
                                 'reason_counts', 'rows_with_multiple_v_genes', 'rows_with_multiple_j_genes',
                                 'source_unchanged_after_read', 'full_vdj_functionality_verified'}
        safe_audit = {'samples': {key: {k: v for k, v in sample.items() if k in allowed_sample_fields}
                                   for key, sample in audit.get('samples', {}).items()},
                      'policies': audit.get('policies', {}), 'limitations': audit.get('limitations', [])}
        # A view must identify its selection. Never silently show a different Top N.
        matching_views = [view for view in views if view['metadata'].get('selection_id') == selection_id
                          or Path(str(view['metadata'].get('selection_dir', ''))).name == selection_id
                          or view['metadata'].get('selection_relative_path') == selection_id]
        projections.sort(key=lambda item: (str(item['metadata'].get('created_at_utc', '')), item['id']), reverse=True)
        matching_views.sort(key=lambda item: (str(item['metadata'].get('created_at_utc', '')), item['path']), reverse=True)
        return {'id': identifier,
                'metadata': {'status': metadata.get('status'), 'subject': metadata.get('subject'),
                             'created_at': metadata.get('created_at_utc'), 'parameters': metadata.get('parameters', {}),
                             'elapsed_seconds': metadata.get('elapsed_seconds'),
                             'notes': [metadata.get('validation_status'), metadata.get('paper_equivalence')]},
                'input_audit': safe_audit, 'selection': {'id': selection_id, 'summary': summary},
                'candidates': candidates, 'selections': selections, 'projections': projections,
                'latest_view': matching_views[0] if matching_views else None}

    def download_path(self, identifier: str, relative: str) -> Path:
        run = self.run_path(identifier)
        path = confined(run, relative)
        parts = relative.split('/')
        if path.name not in DOWNLOAD_NAMES or not path.is_file():
            raise ApiError(403, 'このファイルは画面から取得できません。')
        if len(parts) == 1:
            valid = path.name in {'run_metadata.json', 'input_audit.json', 'cluster_scores.csv'}
        elif parts[0].startswith('selection_'):
            valid = len(parts) == 2 and path.name in {'candidates.csv', 'candidate_provenance.json',
                                                     'selection_summary.json', 'selection_source.json'}
        elif parts[0].startswith('projection_'):
            valid = path.name in {'projection_metadata.json', 'view_metadata.json', 'comparison.png'} and len(parts) <= 4
        else:
            valid = False
        if not valid:
            raise ApiError(403, 'このファイルは画面から取得できません。')
        return path

    def submit(self, kind: str, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ApiError(400, '送信形式が不正です。')
        with self.lock:
            current = self.job()
            if ((self.process is not None and self.process.poll() is None)
                    or (current is not None and current.get('status') in ('queued', 'running'))):
                raise ApiError(409, '処理中です。完了してから次の処理を開始してください。')
            decoded = None
            if kind == 'analyze':
                request = analysis_parameters(data)
                decoded = decode_uploads(data.get('files'))
            elif kind in ('visualize', 'reselect'):
                run = self.run_path(data.get('run_id'))
                if read_json(confined(run, 'run_metadata.json')).get('status') != 'completed':
                    raise ApiError(400, '完了した解析だけを選択してください。')
                request = {'run_id': run.name}
                if kind == 'reselect':
                    request['top_n'] = positive_int(data.get('top_n'), 'top_n')
                else:
                    selected = data.get('selection_id', 'selection_initial')
                    request['selection_id'] = self.child_path(run, selected, 'selection_').name
                if data.get('projection_id') is not None:
                    request['projection_id'] = self.child_path(run, data['projection_id'], 'projection_').name
            else:
                raise ApiError(404, 'この操作は利用できません。')
            # Re-check directory boundaries at every write, including new symlinks.
            for parent in (self.jobs, self.inputs):
                if self.root not in parent.resolve().parents:
                    raise ApiError(403, '保存先の場所を確認してください。')
                parent.mkdir(parents=True, exist_ok=True)
            job_id = 'job_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ_') + uuid.uuid4().hex[:8]
            folder = self.jobs / job_id
            protect_folder(folder)
            if decoded is not None:
                input_folder = self.inputs / job_id
                protect_folder(input_folder)
                manifest = {'created_at': utc_now(), 'files': {}}
                for phase, item in decoded.items():
                    with (input_folder / f'{phase}.csv').open('xb') as handle:
                        handle.write(item['bytes'])
                    manifest['files'][phase] = {'original_name': item['name'], 'copy_name': f'{phase}.csv',
                                                'bytes': len(item['bytes']),
                                                'sha256': hashlib.sha256(item['bytes']).hexdigest()}
                write_atomic_json(input_folder / 'upload_manifest.json', manifest)
                request['input_id'] = job_id
            now = utc_now()
            job = {'id': job_id, 'type': kind, 'status': 'queued', 'stage': 'queued', 'progress': None,
                   'created_at': now, 'updated_at': now, 'result': None, 'error': None}
            write_atomic_json(folder / 'request.json', {'type': kind, 'parameters': request})
            write_atomic_json(folder / 'job.json', job)
            self.job_path = folder / 'job.json'
            try:
                self.process = self.spawn([sys.executable, str(self.root / 'scripts/app_worker.py'),
                                           '--job-id', job_id], cwd=self.root,
                                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                write_atomic_json(folder / 'process.json', {'pid': self.process.pid, 'created_at': utc_now()})
            except OSError:
                job.update(status='failed', stage='failed', error='処理を開始できませんでした。起動環境を確認してください。',
                           updated_at=utc_now())
                write_atomic_json(self.job_path, job)
                raise ApiError(500, job['error']) from None
            return job


class LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, app: Application, port: int = 8765):
        self.app = app
        records = app.root / 'local_records'
        if app.root not in records.resolve().parents:
            raise ValueError('Private records must remain in the project.')
        records.mkdir(exist_ok=True)
        self._instance_lock = (records / 'ui_server.lock').open('a+b')
        self._instance_lock.seek(0, 2)
        if self._instance_lock.tell() == 0:
            self._instance_lock.write(b'1')
            self._instance_lock.flush()
        self._instance_lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self._instance_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._instance_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._instance_lock.close()
            raise OSError('An interface for this project is already running.') from None
        try:
            super().__init__(('127.0.0.1', port), Handler)
        except BaseException:
            self._release_instance_lock()
            raise
        self.origin = f'http://127.0.0.1:{self.server_port}'

    def _release_instance_lock(self):
        if self._instance_lock.closed:
            return
        self._instance_lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self._instance_lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._instance_lock.fileno(), fcntl.LOCK_UN)
        self._instance_lock.close()

    def server_close(self):
        super().server_close()
        self._release_instance_lock()


class Handler(BaseHTTPRequestHandler):
    server_version = 'LM-QASAS'

    def log_message(self, format, *args):
        # URLs may contain local identifiers. Do not log requests to the console.
        pass

    def _check_request(self, *, mutation: bool = False):
        if self.client_address[0] != '127.0.0.1' or self.headers.get('Host') != self.server.origin[7:]:
            raise ApiError(403, 'このPCの127.0.0.1から開いてください。')
        origin = self.headers.get('Origin')
        if origin is not None and origin != self.server.origin:
            raise ApiError(403, '外部のページからの操作は受け付けません。')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise ApiError(403, '外部のページからの操作は受け付けません。')
        if mutation and not hmac.compare_digest(self.headers.get('X-LMQASAS-Token', '').encode('utf-8'), self.server.app.token.encode('utf-8')):
            raise ApiError(403, '画面を再読み込みしてから操作してください。')
        target = urlsplit(self.path)
        if target.scheme or target.netloc:
            raise ApiError(400, '要求の形式が不正です。')
        query = parse_qs(target.query, keep_blank_values=True)
        if any(len(values) != 1 for values in query.values()):
            raise ApiError(400, '要求の形式が不正です。')
        return target.path, {key: value[0] for key, value in query.items()}

    def _reply(self, status: int, body: bytes, content_type: str, *, filename: str | None = None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value):
        self._reply(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'), 'application/json; charset=utf-8')

    def do_GET(self):
        try:
            path, query = self._check_request()
            app = self.server.app
            if path == '/api/bootstrap':
                self._json(200, {'token': app.token})
            elif path == '/api/state':
                self._json(200, {'runs': app.list_runs(), 'job': app.job()})
            elif path == '/api/run':
                self._json(200, app.run_detail(query.get('run'), query.get('selection')))
            elif path == '/api/file':
                file = app.download_path(query.get('run'), query.get('path'))
                mime = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
                filename = file.name if file.suffix != '.png' or query.get('download') == '1' else None
                self._reply(200, file.read_bytes(), mime, filename=filename)
            elif path in ('/', '/index.html', '/app.js', '/style.css'):
                file = confined(app.root / 'app', 'index.html' if path in ('/', '/index.html') else path[1:])
                mime = {'html': 'text/html', 'js': 'application/javascript', 'css': 'text/css'}[file.suffix[1:]]
                self._reply(200, file.read_bytes(), mime + '; charset=utf-8')
            else:
                raise ApiError(404, '指定したページはありません。')
        except ApiError as exc:
            self._json(exc.status, {'error': exc.message})
        except Exception:
            self._json(500, {'error': '保存した結果を読み込めませんでした。結果の記録を確認してください。'})

    def do_POST(self):
        try:
            path, _ = self._check_request(mutation=True)
            if path not in ('/api/analyze', '/api/visualize', '/api/reselect'):
                raise ApiError(404, 'この操作は利用できません。')
            if self.headers.get('Transfer-Encoding') is not None:
                raise ApiError(400, '送信形式が不正です。')
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                raise ApiError(415, 'JSON形式で送信してください。')
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                raise ApiError(400, '送信サイズが不正です。') from None
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ApiError(413, '送信サイズが上限を超えています。')
            self.connection.settimeout(30)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ApiError(400, '送信が完了していません。')
            try:
                data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            except (ValueError, UnicodeError):
                raise ApiError(400, '送信形式が不正です。') from None
            self._json(202, {'job': self.server.app.submit(path.rsplit('/', 1)[1], data)})
        except ApiError as exc:
            self._json(exc.status, {'error': exc.message})
        except Exception:
            self._json(500, {'error': '処理を開始できませんでした。入力と保存先を確認してください。'})
