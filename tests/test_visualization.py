"""Synthetic-only visualization checks; no research sequences or checkpoint."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lmqasas.checkpoint import digest
from lmqasas.pipeline import reselect, save_selection, write_json
from lmqasas.visualization import (PHASES, SOURCE_FILES, _density_grid,
                                   create_projection, render_selection)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob('*') if p.is_file()}


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'synthetic_run'
        self.run.mkdir()
        self.sequences = ['CAAAF', 'CACAF', 'CADAF', 'CAEAF', 'CAFAF', 'CAGAF', 'CAHAF', 'CAIAF']
        self.vectors = np.random.default_rng(1234).normal(size=(8, 480))
        with (self.run / 'embeddings.npy').open('xb') as stream:
            np.save(stream, self.vectors, allow_pickle=False)
        write_json(self.run / 'embedding_sequences.json', self.sequences)
        self.clones = []
        for phase, indices in zip(PHASES, ([0, 1, 2, 0, 3], [0, 1, 2, 4, 5, 6], [0, 2, 6, 7])):
            for sequence_index in indices:
                index = len(self.clones)
                self.clones.append({'clone_id': index, 'embedding_index': sequence_index,
                                    'cdr3': self.sequences[sequence_index], 'timepoint': phase,
                                    'source_row': index + 2, 'subject': 'synthetic_only',
                                    'v_gene': 'IGHV1-2', 'j_gene': 'IGHJ4', 'isotype': 'IGHG1'})
        write_json(self.run / 'clones.json', self.clones)
        peak = [{**c, 'score': float(20 - c['embedding_index']), 'cluster_id': c['embedding_index']}
                for c in self.clones if c['timepoint'] == 'Peak']
        write_json(self.run / 'scored_peak.json', peak)
        summary = save_selection(peak, 1, self.run / 'selection_initial')
        metadata = {'status': 'completed', 'input_paths': {p: str(self.root / 'inputs' / f'{p}.csv') for p in PHASES},
                    'inputs_unchanged_after_run': {p: True for p in PHASES}, 'selection': summary,
                    'artifact_sha256': snapshot(self.run)}
        write_json(self.run / 'run_metadata.json', metadata)
        self.before = snapshot(self.run)

    def tearDown(self):
        self.temp.cleanup()

    def fake_fit(self, vectors, settings):
        self.assertEqual(len(vectors), len(self.clones))
        np.testing.assert_array_equal(vectors, np.asarray(self.vectors[[c['embedding_index'] for c in self.clones]], dtype=np.float32))
        np.testing.assert_array_equal(vectors[0], vectors[3])
        self.assertEqual(settings['unique'], False)
        self.assertEqual(settings['low_memory'], True)
        self.assertEqual(settings['n_jobs'], 1)
        self.assertEqual(settings['n_neighbors'], 3)
        self.assertEqual(settings['metric'], 'cosine')
        return np.column_stack((np.arange(len(vectors)), np.sin(np.arange(len(vectors))))).astype(np.float32)

    def projection(self):
        with patch('lmqasas.visualization._fit_umap', side_effect=self.fake_fit) as fitted:
            projection = create_projection(self.run, n_neighbors=3, n_epochs=20)
        fitted.assert_called_once()
        return projection

    def assert_run_unchanged(self):
        self.assertEqual(self.before, {name: digest(self.run / name) for name in self.before})

    def test_projection_retains_all_clone_points_and_source_provenance(self):
        projection = self.projection()
        metadata = read_json(projection / 'projection_metadata.json')
        self.assertEqual(metadata['status'], 'completed')
        self.assertEqual(metadata['observations'], 15)
        self.assertEqual(metadata['timepoint_counts'], {'Pre': 5, 'Peak': 6, 'Post': 4})
        self.assertFalse(metadata['downsampling'])
        self.assertFalse(metadata['sequence_deduplication_of_observations'])
        self.assertEqual(set(metadata['source_sha256']), set(SOURCE_FILES) | {'run_metadata.json'})
        self.assertEqual(np.load(projection / 'coordinates.npy', allow_pickle=False).shape, (15, 2))
        self.assertIn('*', (projection / '.gitignore').read_text().splitlines())
        self.assert_run_unchanged()

    def test_density_uses_shared_axes_bandwidth_and_unit_integral_per_phase(self):
        coords = np.array([[0, 0], [1, 1], [2, 0], [3, 2], [5, 0], [6, 1]], dtype=float)
        x, y, densities, metadata = _density_grid(coords, ['Pre', 'Pre', 'Peak', 'Peak', 'Post', 'Post'])
        self.assertEqual(densities.shape, (3, 96, 96))
        self.assertLess(x.min(), coords[:, 0].min())
        self.assertGreater(x.max(), coords[:, 0].max())
        self.assertLess(y.min(), coords[:, 1].min())
        self.assertGreater(y.max(), coords[:, 1].max())
        np.testing.assert_allclose(np.trapezoid(np.trapezoid(densities, x=x, axis=2), x=y, axis=1), 1, atol=1e-12)
        expected = np.sqrt(np.mean(np.var(coords, axis=0, ddof=1))) * len(coords) ** (-1/6)
        self.assertAlmostEqual(metadata['bandwidth'], expected)

    def test_selection_overlay_matches_all_timepoints_and_reuses_projection(self):
        projection = self.projection()
        projection_before = snapshot(projection)
        selected_three = reselect(self.run, 3)
        with patch('lmqasas.visualization._fit_umap', side_effect=AssertionError('No refitting allowed')):
            first = render_selection(self.run, projection)
            second = render_selection(self.run, projection, selected_three)
        one = read_json(first / 'view_metadata.json')
        three = read_json(second / 'view_metadata.json')
        self.assertEqual(one['status'], 'completed')
        self.assertEqual(one['candidate_count'], 1)
        self.assertEqual(one['selected_clone_ids'], {'Pre': [0, 3], 'Peak': [5], 'Post': [11]})
        self.assertEqual(one['selected_observation_counts'], {'Pre': 2, 'Peak': 1, 'Post': 1})
        self.assertEqual(one['selected_cdr3_type_counts'], {'Pre': 1, 'Peak': 1, 'Post': 1})
        self.assertEqual(three['candidate_count'], 3)
        self.assertEqual(three['selection_id'], selected_three.name)
        self.assertEqual(three['projection_id'], projection.name)
        self.assertEqual(three['selected_clone_ids'], {'Pre': [0, 1, 2, 3], 'Peak': [5, 6, 7], 'Post': [11, 12]})
        self.assertEqual(one['display'], three['display'])
        self.assertTrue(three['projection_reused_without_refit'])
        for folder in (first, second):
            self.assertGreater((folder / 'comparison.png').stat().st_size, 1000)
            self.assertEqual((folder / 'comparison.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(projection_before, {name: digest(projection / name) for name in projection_before})
        self.assert_run_unchanged()

    def test_changed_source_hash_is_rejected_before_projection_creation(self):
        with (self.run / 'embeddings.npy').open('ab') as stream:
            stream.write(b'synthetic_tampering')
        before = set(self.run.iterdir())
        with patch('lmqasas.visualization._fit_umap') as fitted:
            with self.assertRaisesRegex(ValueError, 'integrity'):
                create_projection(self.run, n_neighbors=3)
        fitted.assert_not_called()
        self.assertEqual(before, set(self.run.iterdir()))

    def test_invalid_clone_mapping_rejected_even_when_file_hash_matches(self):
        self.clones[0]['embedding_index'] = 3
        path = self.run / 'clones.json'
        path.write_text(json.dumps(self.clones), encoding='utf-8')
        metadata_path = self.run / 'run_metadata.json'
        metadata = read_json(metadata_path)
        metadata['artifact_sha256']['clones.json'] = digest(path)
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'correspondence'):
            create_projection(self.run, n_neighbors=3)

    def test_invalid_parameters_and_too_many_neighbors_are_not_silently_adjusted(self):
        before = set(self.run.iterdir())
        for options in ({'n_neighbors': 15}, {'n_neighbors': 1}, {'n_neighbors': True},
                        {'min_dist': float('nan')}, {'min_dist': 2}, {'n_epochs': 0},
                        {'seed': -1}, {'metric': 'euclidean'}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    create_projection(self.run, **options)
        self.assertEqual(before, set(self.run.iterdir()))

    def test_failed_projection_is_recorded_and_not_rendered(self):
        with patch('lmqasas.visualization._fit_umap', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaises(RuntimeError):
                create_projection(self.run, n_neighbors=3)
        projection = next(self.run.glob('projection_*'))
        self.assertEqual(read_json(projection / 'projection_metadata.json')['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            render_selection(self.run, projection)
        self.assertFalse(list(projection.glob('view_*')))

    def test_source_changed_during_projection_marks_output_failed(self):
        def tampering_fit(vectors, settings):
            points = self.fake_fit(vectors, settings)
            # Only a generated temporary fixture is changed in this test.
            with (self.run / 'scored_peak.json').open('a', encoding='utf-8') as stream:
                stream.write(' ')
            return points
        with patch('lmqasas.visualization._fit_umap', side_effect=tampering_fit):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                create_projection(self.run, n_neighbors=3)
        projection = next(self.run.glob('projection_*'))
        self.assertEqual(read_json(projection / 'projection_metadata.json')['status'], 'failed')

    def test_changed_projection_points_are_rejected_before_rendering(self):
        projection = self.projection()
        with (projection / 'coordinates.npy').open('ab') as stream:
            stream.write(b'synthetic_tampering')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            render_selection(self.run, projection)
        self.assertFalse(list(projection.glob('view_*')))

    def test_reselection_csv_tampering_rejected_by_score_based_recomputation(self):
        projection = self.projection()
        selection = reselect(self.run, 3)
        path = selection / 'candidates.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]['cdr3'] = 'CWWWF'
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'Candidate CSV'):
            render_selection(self.run, projection, selection)
        self.assertFalse(list(projection.glob('view_*')))

    def test_selection_provenance_tampering_is_rejected(self):
        projection = self.projection()
        selection = reselect(self.run, 3)
        path = selection / 'candidate_provenance.json'
        provenance = read_json(path)
        provenance[0]['clone_records'][0]['source_row'] = 999
        path.write_text(json.dumps(provenance), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            render_selection(self.run, projection, selection)

    def test_unfinished_run_is_rejected_without_outputs(self):
        path = self.run / 'run_metadata.json'
        metadata = read_json(path)
        metadata['status'] = 'failed'
        path.write_text(json.dumps(metadata), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Only completed'):
            create_projection(self.run, n_neighbors=3)
        self.assertFalse(list(self.run.glob('projection_*')))

    def test_real_umap_reproducible_small_synthetic_run(self):
        # Runs the actual algorithm; the vectors are random fixtures, not AbLang2.
        first = create_projection(self.run, n_neighbors=3, n_epochs=20, seed=17)
        second = create_projection(self.run, n_neighbors=3, n_epochs=20, seed=17)
        a = np.load(first / 'coordinates.npy', allow_pickle=False)
        b = np.load(second / 'coordinates.npy', allow_pickle=False)
        self.assertEqual(a.shape, (15, 2))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a, b)
        self.assert_run_unchanged()


if __name__ == '__main__':
    unittest.main()
