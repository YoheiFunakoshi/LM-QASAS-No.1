"""Synthetic-only HTTP, upload, and result-boundary tests; no research data."""
import base64
import csv
import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from lmqasas.server import (ApiError, Application, LocalHTTPServer, analysis_parameters,
                            confined, decode_uploads, read_json, resolve_uploaded_inputs,
                            write_atomic_json)


class FakeProcess:
    pid = 99999999

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


def synthetic_uploads():
    text = ('Vseg,Jseg,CDR3,AAlength,NTlength,Type,Cseg,Counts,Frequency(%)\n'
            'IGHV1-1,IGHJ1,CASSF,5,15,WithConserved_NoStop,IGHG1,1,0.1\n')
    content = base64.b64encode(text.encode('utf-8')).decode('ascii')
    return {phase: {'name': phase + '.csv', 'base64': content} for phase in ('Pre', 'Peak', 'Post')}


def synthetic_excel_uploads():
    # This factory writes only invented report records with standard-library OOXML.
    from xlsx_fixture import synthetic_xlsx
    content = base64.b64encode(synthetic_xlsx()).decode('ascii')
    return {phase: {'name': phase + '.XLSX', 'base64': content} for phase in ('Pre', 'Peak', 'Post')}


def synthetic_run(root):
    run = root / 'outputs/run_synthetic'
    selection = run / 'selection_initial'
    selection.mkdir(parents=True)
    summary = {'requested': 300, 'available': 1, 'returned': 1, 'shortfall': 299, 'boundary_tie': None}
    write_atomic_json(run / 'run_metadata.json', {'created_at_utc': '2026-01-01T00:00:00Z',
                                               'subject': 'SYNTHETIC', 'status': 'completed',
                                               'parameters': {'top_n': 300}, 'selection': summary,
                                               'input_paths': {'Pre': 'private/path.csv'}})
    write_atomic_json(run / 'input_audit.json', {'samples': {'Pre': {'total_rows': 1, 'clone_count': 1,
                                                                  'input_format': 'takara_rg_xlsx',
                                                                  'source_sheet': 'Back_data',
                                                                  'isotype_granularity': 'subclass',
                                                                  'cdr3_definition': 'as_reported_no_boundary_repair',
                                                                  'source_file': 'private/path.csv',
                                                                  'exclusions': [{'source_row': 2}]}}})
    write_atomic_json(selection / 'selection_summary.json', summary)
    (selection / 'candidates.csv').write_text('rank,cdr3,score,clone_count,cluster_ids\n1,CASSF,2.0,1,0\n', encoding='utf-8')
    (run / 'embeddings.npy').write_bytes(b'synthetic-not-real-embedding')
    (run / 'clones.json').write_text('[]', encoding='utf-8')
    return run


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'app').mkdir()
        (self.root / 'app/index.html').write_text('<html>Synthetic</html>', encoding='utf-8')
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/app_worker.py').write_text('# synthetic stub', encoding='utf-8')
        self.process = FakeProcess()
        self.spawn_calls = []

        def spawn(*args, **kwargs):
            self.spawn_calls.append((args, kwargs))
            return self.process

        self.app = Application(self.root, spawn=spawn)
        self.server = LocalHTTPServer(self.app, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, method, path, data=None, headers=None):
        headers = dict(headers or {})
        if data is not None:
            data = json.dumps(data).encode('utf-8')
            headers.setdefault('Content-Type', 'application/json')
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        client.request(method, path, data, headers)
        response = client.getresponse()
        body, status, output_headers = response.read(), response.status, dict(response.getheaders())
        client.close()
        if output_headers.get('Content-Type', '').startswith('application/json'):
            body = json.loads(body)
        return status, body, output_headers

    def post(self, path, data, **headers):
        return self.request('POST', path, data, {'X-LMQASAS-Token': self.app.token, **headers})

    def test_bootstrap_and_static_have_security_headers(self):
        status, body, headers = self.request('GET', '/api/bootstrap')
        self.assertEqual(status, 200)
        self.assertEqual(body['token'], self.app.token)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        self.assertEqual(self.request('GET', '/')[0], 200)

    def test_second_server_for_same_project_is_rejected(self):
        with self.assertRaises(OSError):
            LocalHTTPServer(Application(self.root), 0)

    def test_requests_reject_external_host_and_origin(self):
        for headers in ({'Host': 'evil.invalid'}, {'Origin': 'https://evil.invalid'},
                        {'Sec-Fetch-Site': 'cross-site'}):
            self.assertEqual(self.request('GET', '/api/bootstrap', headers=headers)[0], 403)
        self.assertEqual(self.post('/api/analyze', {}, Origin='null')[0], 403)

    def test_post_requires_token_and_accepts_exact_origin(self):
        self.assertEqual(self.request('POST', '/api/analyze', {})[0], 403)
        status, body, _ = self.post('/api/analyze', {'subject': 'SYNTHETIC', 'files': synthetic_uploads()},
                                    Origin=self.server.origin)
        self.assertEqual(status, 202)
        self.assertEqual(body['job']['status'], 'queued')

    def test_upload_creates_copies_and_manifest_and_sanitized_job(self):
        source = synthetic_uploads()
        job = self.app.submit('analyze', {'subject': 'SYNTHETIC', 'files': source})
        copied = self.root / 'local_records/ui_inputs' / job['id']
        self.assertEqual((copied / 'Pre.csv').read_bytes(), base64.b64decode(source['Pre']['base64']))
        manifest = read_json(copied / 'upload_manifest.json')
        self.assertEqual(manifest['files']['Pre']['original_name'], 'Pre.csv')
        self.assertEqual(manifest['files']['Pre']['copy_name'], 'Pre.csv')
        self.assertEqual(manifest['files']['Pre']['sha256'], hashlib.sha256((copied / 'Pre.csv').read_bytes()).hexdigest())
        self.assertTrue((copied / '.gitignore').exists())
        self.assertNotIn('files', job)
        self.assertNotIn('subject', job)
        self.assertEqual(len(self.spawn_calls), 1)
        resolved = resolve_uploaded_inputs(copied)
        self.assertEqual({phase: path.name for phase, path in resolved.items()},
                         {phase: phase + '.csv' for phase in ('Pre', 'Peak', 'Post')})

    def test_worker_rejects_ambiguous_or_changed_private_copies(self):
        job = self.app.submit('analyze', {'subject': 'SYNTHETIC', 'files': synthetic_uploads()})
        copied = self.app.inputs / job['id']
        extra = copied / 'Pre.xlsx'
        extra.write_bytes(b'synthetic second file')
        with self.assertRaises(ApiError):
            resolve_uploaded_inputs(copied)
        extra.unlink()
        original = (copied / 'Pre.csv').read_bytes()
        (copied / 'Pre.csv').write_bytes(original.replace(b'CASSF', b'CATSF'))
        with self.assertRaises(ApiError):
            resolve_uploaded_inputs(copied)

    def test_excel_upload_preserves_bytes_and_suffix_and_worker_resolves_manifest(self):
        files = synthetic_excel_uploads()
        status, body, _ = self.post('/api/analyze', {'subject': 'SYNTHETIC', 'files': files})
        self.assertEqual(status, 202)
        copied = self.app.inputs / body['job']['id']
        manifest = read_json(copied / 'upload_manifest.json')
        paths = resolve_uploaded_inputs(copied)
        self.assertEqual(set(paths), {'Pre', 'Peak', 'Post'})
        for phase, path in paths.items():
            with self.subTest(phase=phase):
                self.assertEqual(path.name, phase + '.xlsx')
                self.assertEqual(path.read_bytes(), base64.b64decode(files[phase]['base64']))
                self.assertEqual(manifest['files'][phase]['original_name'], phase + '.XLSX')
                self.assertEqual(manifest['files'][phase]['copy_name'], phase + '.xlsx')
                self.assertEqual(manifest['files'][phase]['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertFalse((copied / (phase + '.csv')).exists())

    def test_excel_and_csv_content_are_not_silently_reinterpreted(self):
        csv_files, excel_files = synthetic_uploads(), synthetic_excel_uploads()
        for data, false_suffix in ((csv_files, '.xlsx'), (excel_files, '.csv')):
            files = {phase: {**item, 'name': phase + false_suffix} for phase, item in data.items()}
            with self.subTest(false_suffix=false_suffix):
                status, _, _ = self.post('/api/analyze', {'subject': 'SYNTHETIC', 'files': files})
                self.assertEqual(status, 400)
        self.assertFalse(self.app.inputs.exists())
        self.assertFalse(self.spawn_calls)

    def test_mixed_csv_and_excel_rejected_before_upload_writes(self):
        files = synthetic_excel_uploads()
        files['Pre'] = synthetic_uploads()['Pre']
        status, body, _ = self.post('/api/analyze', {'subject': 'SYNTHETIC', 'files': files})
        self.assertEqual(status, 400)
        self.assertIn('混在', body['error'])
        self.assertFalse(self.app.inputs.exists())
        self.assertFalse(self.spawn_calls)

    def test_worker_rejects_manifest_paths_wrong_phases_and_unknown_extensions(self):
        job = self.app.submit('analyze', {'subject': 'SYNTHETIC', 'files': synthetic_uploads()})
        copied = self.app.inputs / job['id']
        manifest_path = copied / 'upload_manifest.json'
        manifest = read_json(manifest_path)
        for name in ('../Pre.csv', 'Peak.csv', 'Pre.xls', 'Pre.csv/child', 'C:\\private.csv'):
            manifest['files']['Pre']['copy_name'] = name
            write_atomic_json(manifest_path, manifest)
            with self.subTest(copy_name=name), self.assertRaises(ApiError):
                resolve_uploaded_inputs(copied)
        manifest['files'].pop('Post')
        write_atomic_json(manifest_path, manifest)
        with self.assertRaises(ApiError):
            resolve_uploaded_inputs(copied)

    def test_worker_supports_legacy_csv_jobs_without_manifest(self):
        copied = self.root / 'legacy'
        copied.mkdir()
        for phase, item in synthetic_uploads().items():
            (copied / (phase + '.csv')).write_bytes(base64.b64decode(item['base64']))
        resolved = resolve_uploaded_inputs(copied)
        self.assertEqual(set(resolved), {'Pre', 'Peak', 'Post'})
        (copied / 'Pre.csv').rename(copied / 'Pre.xlsx')
        with self.assertRaises(ApiError):
            resolve_uploaded_inputs(copied)

    def test_second_submission_is_busy_and_does_not_copy_more_files(self):
        data = {'subject': 'SYNTHETIC', 'files': synthetic_uploads()}
        self.app.submit('analyze', data)
        status, _, _ = self.post('/api/analyze', data)
        self.assertEqual(status, 409)
        self.assertEqual(len(list(self.app.inputs.iterdir())), 1)
        self.assertEqual(len(self.spawn_calls), 1)

    def test_worker_exit_without_completion_is_failure(self):
        self.app.submit('analyze', {'subject': 'SYNTHETIC', 'files': synthetic_uploads()})
        self.process.returncode = 2
        state = self.app.job()
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['stage'], 'interrupted')
        self.assertNotIn('CASSF', json.dumps(state))

    def test_saved_d_policy_is_exposed_without_relabeling_or_private_provenance(self):
        run = synthetic_run(self.root)
        for policy in ('takara-rg-hIGH20181210-v1', 'takara-rg-hIGH20181210-v2-ignore-d'):
            audit = read_json(run / 'input_audit.json')
            audit['samples']['Pre']['input_policy'] = policy
            write_atomic_json(run / 'input_audit.json', audit)
            before = (run / 'input_audit.json').read_bytes()
            detail = self.app.run_detail(run.name)
            sample = detail['input_audit']['samples']['Pre']
            self.assertEqual(sample['input_policy'], policy)
            self.assertNotIn('source_file', sample)
            self.assertNotIn('raw_records', sample)
            self.assertEqual((run / 'input_audit.json').read_bytes(), before)

    def test_worker_completion_race_is_not_overwritten(self):
        self.app.submit('analyze', {'subject': 'SYNTHETIC', 'files': synthetic_uploads()})

        def complete_on_poll():
            final = read_json(self.app.job_path)
            final.update(status='completed', stage='completed')
            write_atomic_json(self.app.job_path, final)
            return 0

        self.process.poll = complete_on_poll
        self.assertEqual(self.app.job()['status'], 'completed')
        self.assertEqual(read_json(self.app.job_path)['status'], 'completed')

    def test_restart_observes_existing_worker_without_overwriting_job(self):
        self.app.submit('analyze', {'subject': 'SYNTHETIC', 'files': synthetic_uploads()})
        before = self.app.job_path.read_bytes()
        with patch('lmqasas.server.process_alive', return_value=True):
            restarted = Application(self.root)
            self.assertEqual(restarted.job()['status'], 'queued')
            with self.assertRaises(ApiError) as error:
                restarted.submit('analyze', {'subject': 'SYNTHETIC', 'files': synthetic_uploads()})
            self.assertEqual(error.exception.status, 409)
        self.assertEqual(self.app.job_path.read_bytes(), before)

    def test_run_detail_returns_aggregate_audit_and_preserves_outputs(self):
        run = synthetic_run(self.root)
        before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()}
        status, detail, _ = self.request('GET', '/api/run?run=run_synthetic')
        self.assertEqual(status, 200)
        self.assertEqual(detail['candidates'][0]['cdr3'], 'CASSF')
        self.assertEqual(detail['selection']['summary']['shortfall'], 299)
        self.assertNotIn('source_file', detail['input_audit']['samples']['Pre'])
        self.assertNotIn('exclusions', detail['input_audit']['samples']['Pre'])
        self.assertEqual(detail['input_audit']['samples']['Pre']['input_format'], 'takara_rg_xlsx')
        self.assertEqual(detail['input_audit']['samples']['Pre']['source_sheet'], 'Back_data')
        self.assertEqual(detail['input_audit']['samples']['Pre']['isotype_granularity'], 'subclass')
        self.assertEqual(detail['input_audit']['samples']['Pre']['cdr3_definition'], 'as_reported_no_boundary_repair')
        self.assertNotIn('input_paths', detail['metadata'])
        after = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_view_matches_selected_candidate_count(self):
        run = synthetic_run(self.root)
        projection = run / 'projection_synthetic'
        projection.mkdir()
        write_atomic_json(projection / 'projection_metadata.json', {'status': 'completed'})
        for suffix, selection in [('first', 'selection_initial'), ('last', 'selection_other')]:
            view = projection / ('view_' + suffix)
            view.mkdir()
            (view / 'comparison.png').write_bytes(b'fake-png')
            write_atomic_json(view / 'view_metadata.json', {'selection_id': selection, 'status': 'completed'})
        detail = self.app.run_detail('run_synthetic')
        self.assertIn('view_first', detail['latest_view']['path'])
        self.assertEqual(detail['latest_view']['metadata']['selection_id'], 'selection_initial')

    def test_only_completed_projections_and_newest_completed_view_are_returned(self):
        run = synthetic_run(self.root)
        for name, created, status in [('older', '2026-01-01T00:00:00Z', 'completed'),
                                       ('newer', '2026-01-02T00:00:00Z', 'completed'),
                                       ('failed', '2026-01-03T00:00:00Z', 'failed')]:
            projection = run / ('projection_' + name)
            projection.mkdir()
            write_atomic_json(projection / 'projection_metadata.json', {'status': status, 'created_at_utc': created})
            interrupted = projection / 'view_aaa_interrupted'
            interrupted.mkdir()
            (interrupted / 'comparison.png').write_bytes(b'fake-png-without-completion-record')
            for suffix, view_status, view_date in [('good', 'completed', created),
                                                    ('partial', 'failed', '2026-12-31T00:00:00Z')]:
                view = projection / ('view_' + suffix)
                view.mkdir()
                (view / 'comparison.png').write_bytes(b'fake-png')
                write_atomic_json(view / 'view_metadata.json', {'status': view_status, 'selection_id': 'selection_initial',
                                                               'created_at_utc': view_date})
        detail = self.app.run_detail('run_synthetic')
        self.assertEqual([p['id'] for p in detail['projections']], ['projection_newer', 'projection_older'])
        self.assertEqual(detail['latest_view']['path'], 'projection_newer/view_good/comparison.png')

    def test_png_is_inline_by_default_and_download_flag_adds_attachment(self):
        run = synthetic_run(self.root)
        view = run / 'projection_synthetic/view_synthetic'
        view.mkdir(parents=True)
        (view / 'comparison.png').write_bytes(b'fake-png')
        url = '/api/file?run=run_synthetic&path=projection_synthetic/view_synthetic/comparison.png'
        inline_status, _, inline_headers = self.request('GET', url)
        attachment_status, _, attachment_headers = self.request('GET', url + '&download=1')
        self.assertEqual((inline_status, attachment_status), (200, 200))
        self.assertNotIn('Content-Disposition', inline_headers)
        self.assertEqual(attachment_headers['Content-Disposition'], 'attachment; filename="comparison.png"')

    def test_download_allowlist_and_traversal(self):
        synthetic_run(self.root)
        self.assertEqual(self.request('GET', '/api/file?run=run_synthetic&path=selection_initial/candidates.csv')[0], 200)
        for relative in ('embeddings.npy', 'clones.json', '../scripts/app_worker.py',
                         'selection_initial/../../app/index.html', '/etc/passwd', 'C:%5Cprivate'):
            self.assertIn(self.request('GET', '/api/file?run=run_synthetic&path=' + relative)[0], (400, 403))
        self.assertEqual(self.request('GET', '/api/run?run=../app')[0], 400)
        self.assertEqual(self.request('GET', '/api/run?run=run_synthetic&run=run_other')[0], 400)

    def test_visualize_and_reselect_only_completed_run(self):
        run = synthetic_run(self.root)
        metadata = read_json(run / 'run_metadata.json')
        metadata['status'] = 'failed'
        write_atomic_json(run / 'run_metadata.json', metadata)
        self.assertEqual(self.post('/api/visualize', {'run_id': run.name})[0], 400)
        self.assertEqual(self.post('/api/reselect', {'run_id': run.name, 'top_n': 500})[0], 400)
        self.assertFalse(self.app.jobs.exists())

    def test_invalid_parameters_fail_before_upload_writes(self):
        for field, value in [('top_n', True), ('top_n', 0), ('n_clusters', 1.5), ('epsilon', 0),
                             ('epsilon', '1'), ('epsilon', float('inf')), ('seed', -1),
                             ('seed', 2**32), ('n_init', 0), ('threads', 0), ('subject', '\n')]:
            with self.subTest(field=field, value=value), self.assertRaises(ApiError):
                analysis_parameters({'subject': 'SYNTHETIC', field: value})
        self.assertFalse(self.app.jobs.exists())

    def test_upload_rejects_paths_encoding_columns_and_invalid_base64(self):
        for field, value in [('name', '../secret.csv'), ('name', 'file.txt'), ('base64', '???'),
                             ('base64', base64.b64encode(b'wrong,columns\n').decode()),
                             ('base64', base64.b64encode(b'\xff\xfe').decode())]:
            files = synthetic_uploads()
            files['Pre'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ApiError):
                decode_uploads(files)
        with self.assertRaises(ApiError):
            decode_uploads({'Pre': synthetic_uploads()['Pre']})

    def test_size_limit_and_nonjson(self):
        with patch('lmqasas.server.MAX_FILE_BYTES', 10):
            with self.assertRaises(ApiError) as error:
                decode_uploads(synthetic_uploads())
            self.assertEqual(error.exception.status, 413)
        with patch('lmqasas.server.MAX_REQUEST_BYTES', 10):
            self.assertEqual(self.post('/api/analyze', {'subject': 'SYNTHETIC'})[0], 413)
        self.assertEqual(self.request('POST', '/api/analyze', {},
                                     {'X-LMQASAS-Token': self.app.token, 'Content-Type': 'text/plain'})[0], 415)

    def test_symlink_cannot_escape_allowed_directory(self):
        synthetic_run(self.root)
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'candidates.csv').write_text('private synthetic text', encoding='utf-8')
        link = self.app.outputs / 'run_synthetic/selection_external'
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Symlink creation is unavailable on this host.')
        with self.assertRaises(ApiError):
            self.app.download_path('run_synthetic', 'selection_external/candidates.csv')

    def test_resolved_path_boundary_rejects_relocated_file(self):
        # Exercise resolution policy on hosts where creating symlinks needs admin.
        base = self.root / 'outputs'
        outside = self.root / 'outside.csv'
        with patch.object(Path, 'resolve', side_effect=[base, outside]):
            with self.assertRaises(ApiError) as error:
                confined(base, 'candidate.csv')
            self.assertEqual(error.exception.status, 403)


if __name__ == '__main__':
    unittest.main()
