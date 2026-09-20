"""Private background job runner; never prints inputs or exception messages."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lmqasas.server import checked_id, confined, read_json, resolve_uploaded_inputs, utc_now, write_atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    try:
        folder = confined(ROOT / 'local_records/ui_jobs', checked_id(args.job_id, 'job_'))
        job_path = confined(folder, 'job.json')
        job = read_json(job_path)
        request = read_json(confined(folder, 'request.json'))
    except Exception:
        return 1

    def update(stage, fraction=None, **extra):
        job.update(stage=stage, progress=fraction, status='running', updated_at=utc_now(), **extra)
        write_atomic_json(job_path, job)
        with (folder / 'progress.log').open('a', encoding='utf-8', newline='\n') as log:
            # stage names and timestamps only; no uploaded content or arbitrary error.
            log.write(json.dumps({'at': job['updated_at'], 'stage': stage, 'progress': fraction}) + '\n')

    try:
        update('loading')
        import ablang2  # noqa: F401
        import torch  # noqa: F401
        import umap  # noqa: F401
        import matplotlib.backends.backend_agg  # noqa: F401
        from lmqasas.embeddings import enable_network_guard
        from lmqasas.pipeline import reselect, run_analysis
        from lmqasas.visualization import create_projection, render_selection
        # All numerical imports precede this guard. The HTTP server is a separate process.
        enable_network_guard()
        params = request['parameters']
        kind = request['type']
        last_update = [0.0, None]

        def progress(stage, fraction):
            if stage == 'completed':
                stage = 'analysis_completed'
            now = time.monotonic()
            if stage != last_update[1] or now - last_update[0] >= 1 or fraction == 1:
                update(stage, fraction)
                last_update[:] = [now, stage]

        if kind == 'analyze':
            upload = confined(ROOT / 'local_records/ui_inputs', checked_id(params['input_id'], 'job_'))
            paths = resolve_uploaded_inputs(upload)
            options = {key: params[key] for key in ('top_n', 'n_clusters', 'epsilon', 'seed', 'n_init', 'batch_size', 'threads')}
            run = run_analysis(paths, params['subject'], ROOT / 'models/ABLANG-2-paired', ROOT / 'outputs',
                               **options, progress=progress, network_guard=True)
            selection = run / 'selection_initial'
        else:
            run = confined(ROOT / 'outputs', checked_id(params['run_id'], 'run_'))
            if kind == 'reselect':
                update('reselection')
                selection = reselect(run, params['top_n'])
            elif kind == 'visualize':
                selection = confined(run, checked_id(params['selection_id'], 'selection_'))
            else:
                raise ValueError('Unknown job type')
        result = {'run_id': run.name, 'selection_id': selection.name, 'projection_id': None, 'view_path': None}
        update('projection', result=result)
        if params.get('projection_id'):
            projection = confined(run, checked_id(params['projection_id'], 'projection_'))
        else:
            projection = None
            for candidate in sorted(run.glob('projection_*'), reverse=True):
                try:
                    candidate = confined(run, checked_id(candidate.name, 'projection_'))
                    metadata = read_json(confined(candidate, 'projection_metadata.json'))
                    settings = metadata.get('parameters', {})
                    defaults = {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'cosine',
                                'random_state': 20260919, 'n_epochs': 200}
                    if metadata.get('status') == 'completed' and all(settings.get(k) == v for k, v in defaults.items()):
                        # render_selection verifies source and projection hashes before reuse.
                        projection = candidate
                        break
                except (OSError, ValueError):
                    continue
            if projection is None:
                projection = create_projection(run, n_neighbors=15, min_dist=0.1, metric='cosine',
                                               seed=20260919, n_epochs=200, progress=progress)
        result['projection_id'] = projection.name
        update('rendering', result=result)
        view = render_selection(run, projection, selection_dir=selection)
        result['view_path'] = (view / 'comparison.png').relative_to(run).as_posix()
        update('completed', 1.0, result=result)
        job.update(status='completed', updated_at=utc_now())
        write_atomic_json(job_path, job)
        return 0
    except BaseException as exc:
        # Even library ValueErrors can contain data. Do not expose their text.
        stage = job.get('stage')
        if isinstance(exc, ModuleNotFoundError):
            error = '必要なPythonライブラリが不足しています。日本語解説書の環境準備を確認してください。'
        elif isinstance(exc, FileNotFoundError):
            error = 'モデル・入力または保存結果のファイルが見つかりません。日本語解説書の準備と保存先を確認してください。'
        elif stage in ('loading', 'validated', 'embedding', 'kmeans'):
            error = '入力CSV / Excel・採用行数・Kなどの設定を確認してください。'
        else:
            error = '画像または結果の作成に失敗しました。保存履歴で解析結果を確認してください。'
        job.update(status='failed', stage='failed', progress=None, error=error,
                   error_type=type(exc).__name__, failed_stage=stage, updated_at=utc_now())
        write_atomic_json(job_path, job)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
