"""Artificial-coordinate checks for KDE-only sensitivity; no UMAP/model inference."""
from pathlib import Path
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('LMQASAS_PROJECT_DIR', str(HERE.parent)))
sys.path.insert(0, str(PROJECT / 'scripts'))
SCRIPT = HERE / 'compare_kde_bandwidth.py'
if not SCRIPT.is_file():
    SCRIPT = PROJECT / 'scripts/compare_kde_bandwidth.py'
spec = importlib.util.spec_from_file_location('kde_comparison', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    source, pooling, reference_dir = (root / name for name in ('run', 'pooling', 'reference'))
    for path in (source, pooling, reference_dir):
        path.mkdir()
    projection = source / 'projection'
    projection.mkdir()
    coords = np.random.default_rng(174).normal(size=(30, 2)).astype(np.float32)
    cached = np.ones((15,480))
    module.compute.save_array(pooling / 'all_embeddings.npy', cached)
    module.compute.save_array(projection / 'coordinates.npy', coords)
    params = {'n_neighbors': 15, 'min_dist': .1, 'metric': 'cosine',
              'random_state': 20260919, 'transform_seed': 20260919}
    clones = [{'timepoint': phase} for phase in module.PHASES for _ in range(10)]
    module.write_json(source / 'clones.json', clones)
    module.write_json(source / 'run_metadata.json', {'input_hashes': {}, 'input_paths': {}})
    source_hashes = {p.name: module.digest(p) for p in source.iterdir() if p.is_file()}
    pm = {'parameters': params, 'kde': {'bandwidth': .65}, 'packages': {}}
    module.write_json(projection / 'projection_metadata.json', pm)
    ph = {p.name: module.digest(p) for p in projection.iterdir() if p.is_file()}
    prior = {'status': 'completed', 'all_clones_retained': True, 'downsampling_for_umap': False,
             'source_run': str(source), 'source_projection': str(projection),
             'source_sha256': source_hashes, 'projection_sha256': ph, 'input_hashes': {},
             'packages': {}, 'seeds': [20260919,20260920], 'fits': {}}
    for seed in prior['seeds']:
        name = f'all_seed{seed}_raw.npy'
        module.compute.save_array(pooling / name, coords if seed == 20260919 else coords[:,::-1] + 1)
        prior['fits'][f'all_seed{seed}'] = {
            'raw_coordinates_file': name, 'raw_sha256': module.digest(pooling / name),
            'parameters': params | {'random_state': seed, 'transform_seed': seed},
            'pooling': 'all_nonpadding', 'source_clone_sha256': source_hashes['clones.json'],
            'embedding_sha256': module.digest(pooling / 'all_embeddings.npy')}
    prior['artifact_sha256'] = {p.name: module.digest(p) for p in pooling.iterdir()}
    module.write_json(pooling / 'comparison_metadata.json', prior)
    fig = module.render._new_figure((13.55,3.81))
    for i in range(3):
        ax = fig.add_axes([.04+i*.32,.16,.28,.64])
        ax.imshow(np.arange(100).reshape(10,10), cmap='viridis')
        ax.set_title(module.PHASES[i] + ' synthetic')
    fig.suptitle('ARTIFICIAL REFERENCE / SYNTHETIC TEST ONLY')
    reference = reference_dir / 'reference.png'
    reference.write_bytes(module.render._png_bytes(fig))
    context = (source, {'input_hashes': {}}, clones, [], cached, source_hashes,
               projection, pm, ph, {})
    return context, pooling, reference


class KdeComparisonTests(unittest.TestCase):
    def test_shared_grid_resolves_smallest_bandwidth(self):
        values = {0: np.array([[-3.,-4.],[1.,5.]]), 1: np.array([[6.,-3.],[-2.,2.]])}
        x,y = module.make_grid(values, .65)
        self.assertLessEqual(max(np.diff(x).max(),np.diff(y).max()), .65*.5*.5*(1+1e-12))
        self.assertLessEqual(x[0], -3-3*.65*2)
        self.assertGreaterEqual(y[-1], 5+3*.65*2)
        with self.assertRaises(ValueError):
            module.make_grid(values, 0)

    def test_coordinate_preservation_density_and_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            context, pooling, reference = fixture(root)
            originals = {str(p):module.digest(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(module.compute, 'load_context', return_value=context), \
                 patch.object(module.compute, 'verify_unchanged') as unchanged, \
                 patch.object(module.compute, '_fit_umap', side_effect=AssertionError('UMAP prohibited')) as fit:
                metadata = module.run(pooling, reference, root/'output')
                fit.assert_not_called()
                unchanged.assert_called_once_with(context)
                with self.assertRaises(FileExistsError):
                    module.run(pooling, reference, root/'output')
                with self.assertRaises(ValueError):
                    module.run(pooling, reference, pooling/'nested')
            output = root/'output'
            self.assertEqual(len(metadata['coordinate_sets']),2)
            self.assertEqual(len(list(output.glob('*_raw.npy'))),2)
            self.assertEqual(len(list(output.glob('*_aligned.npy'))),2)
            self.assertEqual(len(metadata['fits']),6)
            for seed, group in metadata['coordinate_sets'].items():
                np.testing.assert_array_equal(np.load(output/group['raw_coordinates_file']),
                                              np.load(pooling/group['pooling_source_file']))
                self.assertEqual(len({f['coordinate_set'] for f in metadata['fits'].values()
                                      if str(f['seed']) == seed}),1)
            axes = None
            for item in metadata['fits'].values():
                x,y,density = module.render._read_grid(output/item['density_file'],metadata['common_kde']['grid_size'])
                if axes is not None:
                    np.testing.assert_array_equal(x,axes[0])
                    np.testing.assert_array_equal(y,axes[1])
                axes = x,y
                self.assertEqual(item['bandwidth'],item['bandwidth_factor']*.65)
                np.testing.assert_allclose(np.trapezoid(np.trapezoid(density,x=x,axis=2),x=y,axis=1),1,rtol=1e-12)
            probes = module.compute.read(output/'gaussian_probes.json')
            self.assertEqual(sum(map(len,probes.values())),72)
            self.assertTrue(all(p['passed'] for probeset in probes.values() for p in probeset))
            self.assertTrue(all(module.digest(Path(p)) == h for p,h in originals.items()))
            self.assertTrue(all(module.digest(output/p) == h for p,h in metadata['artifact_sha256'].items()))
            self.assertEqual(len(list(output.glob('*.png'))),2)

    def test_rejects_corrupt_previous_coordinate_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            _,pooling,_ = fixture(Path(temp))
            with (pooling/'all_seed20260919_raw.npy').open('ab') as handle:
                handle.write(b'corrupt')
            with patch.object(module.compute,'load_context') as context:
                with self.assertRaises(ValueError):
                    module.load_inputs(pooling)
                context.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
