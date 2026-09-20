"""Local CSV/XLSX -> clone -> embedding -> K-means -> candidate workflow."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import uuid

import numpy as np

from .checkpoint import digest
from .embeddings import LocalAbLang2
from .inputs import load_three_inputs
from .scoring import run_kmeans
from .selection import select_unique_cdr3, validate_requested_count

ROOT = Path(__file__).resolve().parents[2]
PHASES = ('Pre', 'Peak', 'Post')


def write_json(path: Path, value) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def protect_output(folder: Path) -> None:
    """Exclude a new result directory even when a custom root is inside docs/."""
    with (folder / '.gitignore').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write('# Local research output: never commit this folder.\n*\n')


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # Text metadata is exported as JSON elsewhere; these columns are numeric
            # or strictly validated amino acid/gene strings, not arbitrary CSV formulas.
            writer.writerow({key: row[key] for key in columns})


def save_selection(records: list[dict], requested: int, folder: Path) -> dict:
    result = select_unique_cdr3(records, requested)
    folder.mkdir(parents=True, exist_ok=False)
    protect_output(folder)
    rows = [{'rank': rank, 'cdr3': candidate.cdr3, 'score': candidate.score,
             'clone_count': len(candidate.clone_records),
             'cluster_ids': ';'.join(str(n) for n in sorted({r['cluster_id'] for r in candidate.clone_records}))}
            for rank, candidate in enumerate(result.candidates, 1)]
    write_csv(folder / 'candidates.csv', rows, ['rank', 'cdr3', 'score', 'clone_count', 'cluster_ids'])
    write_json(folder / 'candidate_provenance.json', [asdict(c) for c in result.candidates])
    summary = {'requested': result.requested, 'available': result.available,
               'returned': result.returned, 'shortfall': result.shortfall,
               'boundary_tie': asdict(result.boundary_tie) if result.boundary_tie else None,
               'unit': 'distinct_CDR_H3_amino_acid_sequence',
               'representative_score': 'max', 'tie_break': 'lexical_CDR3'}
    write_json(folder / 'selection_summary.json', summary)
    return summary


def code_provenance() -> dict:
    try:
        result = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=False)
        commit = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        commit = None
    paths = sorted((ROOT / 'src/lmqasas').glob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    return {'git_base_commit': commit,
            'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths}}


def validate_output_location(output_root: Path, paths: dict[str, Path]) -> Path:
    location = output_root.resolve()
    if any(p in {'LM-QASAS No.1 データ', 'LM-QASAS No.1 資料', '.git', 'models', '.venv'}
           for p in location.parts):
        raise ValueError('Output must be outside protected input, material, and environment folders.')
    for path in paths.values():
        parent = path.resolve().parent
        if location == parent or parent in location.parents:
            raise ValueError('Output must be outside the source CSV directories.')
    return location


def run_analysis(paths: dict[str, Path], subject: str, model_dir: Path, output_root: Path,
                 *, top_n=1000, n_clusters=500, epsilon=1.0, seed=20260919,
                 n_init=10, batch_size=32, threads=4, progress=None,
                 encoder=None, network_guard=False) -> Path:
    """Run one subject; encoder injection is solely for synthetic integration tests."""
    validate_requested_count(top_n)
    for value, name in ((n_clusters, 'clusters'), (n_init, 'n_init'),
                        (batch_size, 'batch_size'), (threads, 'threads')):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32).')
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError('epsilon must be a finite positive real number.')
    try:
        epsilon = float(epsilon)
    except OverflowError:
        raise ValueError('epsilon must be a finite positive real number.') from None
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be a finite positive real number.')
    output_root = validate_output_location(output_root, paths)
    bundle = load_three_inputs(paths, subject)
    sequences = sorted({c['cdr3'] for c in bundle.clones})
    if type(n_clusters) is not int or n_clusters < 1 or n_clusters > len(sequences):
        raise ValueError('Cluster count must not exceed available distinct sequences; choose it explicitly.')
    folder = output_root / ('run_' + utc_stamp() + '_' + uuid.uuid4().hex[:8])
    started = perf_counter()
    metadata = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'subject': subject, 'status': 'running', 'code': code_provenance(),
                'parameters': {'top_n': top_n, 'n_clusters': n_clusters, 'epsilon': epsilon,
                               'seed': seed, 'n_init': n_init, 'batch_size': batch_size, 'threads': threads},
                'input_hashes': bundle.input_hashes,
                'input_format': bundle.audit['policies']['input_format'],
                'input_policy_version': bundle.audit['policy_version'],
                'input_policies': {phase: sample['input_policy'] for phase, sample in bundle.audit['samples'].items()},
                'input_paths': {key: str(Path(path).resolve()) for key, path in paths.items()},
                'python': platform.python_version(), 'network_guard': network_guard,
                'packages': {name: importlib.metadata.version(name) for name in
                             ('numpy', 'torch', 'ablang2', 'scikit-learn', 'scipy', 'threadpoolctl')},
                'validation_status': 'exploratory_candidates_not_antigen_binding_validation',
                'paper_equivalence': 'unconfirmed_provisional_preprocessing_pooling_and_defaults',
                'outputs_private': True}
    if metadata['input_format'] == 'takara_rg_xlsx':
        metadata['packages'].update({name: importlib.metadata.version(name)
                                     for name in ('openpyxl', 'et-xmlfile', 'defusedxml')})
    folder.mkdir(parents=True, exist_ok=False)
    protect_output(folder)
    try:
        write_json(folder / 'input_audit.json', bundle.audit)
        if progress:
            progress('validated', 1.0)
        provider = encoder or LocalAbLang2(model_dir, batch_size=batch_size, threads=threads, seed=seed)
        embeddings = provider.encode(sequences, progress=progress)
        if embeddings.shape != (len(sequences), 480) or not np.isfinite(embeddings).all():
            raise ValueError('Embedding shape must be unique sequence count by 480, with finite values.')
        metadata['embedding'] = provider.metadata
        with (folder / 'embeddings.npy').open('xb') as handle:
            np.save(handle, embeddings, allow_pickle=False)
        write_json(folder / 'embedding_sequences.json', sequences)
        sequence_index = {seq: i for i, seq in enumerate(sequences)}
        for i, clone in enumerate(bundle.clones):
            clone['clone_id'] = i
            clone['embedding_index'] = sequence_index[clone['cdr3']]
        write_json(folder / 'clones.json', bundle.clones)
        if progress:
            progress('kmeans', 0.0)
        clone_embeddings = embeddings[[c['embedding_index'] for c in bundle.clones]]
        scored = run_kmeans(clone_embeddings, [c['timepoint'] for c in bundle.clones],
                            n_clusters=n_clusters, epsilon=epsilon, seed=seed, n_init=n_init, threads=threads)
        metadata['scoring'] = scored.metadata
        peak = []
        for i, clone in enumerate(bundle.clones):
            if clone['timepoint'] == 'Peak':
                peak.append({**clone, 'score': float(scored.scores[i]), 'cluster_id': int(scored.labels[i])})
        with (folder / 'cluster_labels.npy').open('xb') as handle:
            np.save(handle, scored.labels, allow_pickle=False)
        write_csv(folder / 'cluster_scores.csv', scored.clusters, ['cluster_id', 'n_pre', 'n_peak', 'n_post', 'score'])
        write_json(folder / 'scored_peak.json', peak)
        metadata['selection'] = save_selection(peak, top_n, folder / 'selection_initial')
        metadata['artifact_sha256'] = {p.relative_to(folder).as_posix(): digest(p)
                                       for p in folder.rglob('*') if p.is_file()}
        metadata['status'] = 'completed'
    except BaseException as exc:
        metadata['status'] = 'failed'
        metadata['error_type'] = type(exc).__name__
        raise
    finally:
        unchanged = {}
        for phase in PHASES:
            try:
                unchanged[phase] = digest(Path(paths[phase])) == bundle.input_hashes[phase]
            except OSError:
                unchanged[phase] = False
        metadata['inputs_unchanged_after_run'] = unchanged
        metadata['elapsed_seconds'] = perf_counter() - started
        if not all(unchanged.values()):
            metadata['status'] = 'failed'
        write_json(folder / 'run_metadata.json', metadata)
    if metadata['status'] != 'completed':
        raise RuntimeError('Input changed during the run; output is not valid.')
    if progress:
        progress('completed', 1.0)
    return folder


def reselect(run_dir: Path, top_n: int) -> Path:
    validate_requested_count(top_n)
    run_dir = run_dir.resolve(strict=True)
    metadata = json.loads((run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('status') != 'completed':
        raise ValueError('Only completed runs can be reselected.')
    validate_output_location(run_dir, {key: Path(value) for key, value in metadata['input_paths'].items()})
    scores = run_dir / 'scored_peak.json'
    expected = metadata.get('artifact_sha256', {}).get('scored_peak.json')
    if not expected or digest(scores) != expected:
        raise ValueError('Stored candidate scores failed the integrity check.')
    records = json.loads(scores.read_text(encoding='utf-8'))
    folder = run_dir / ('selection_' + str(top_n) + '_' + utc_stamp() + '_' + uuid.uuid4().hex[:8])
    save_selection(records, top_n, folder)
    write_json(folder / 'selection_source.json', {'source_run': str(run_dir), 'source_score_sha256': expected,
                                               'code': code_provenance()})
    return folder
