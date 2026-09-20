"""Compare two means of identical AbLang2 hidden states, keeping all clone rows.

This is an offline sensitivity analysis, not a production policy change. Inputs,
model weights and existing runs are read-only; every output remains local.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from lmqasas.checkpoint import digest
from lmqasas.embeddings import LocalAbLang2, enable_network_guard
from lmqasas.pipeline import PHASES, code_provenance, protect_output, validate_output_location, write_json
from lmqasas.visualization import _load_run, _verify_sources, _fit_umap


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_array(path, value):
    with Path(path).open('xb') as handle:
        np.save(handle, value, allow_pickle=False)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite metric.')
    return {k: float(v) for k, v in zip(
        ('min', 'p05', 'median', 'p95', 'max', 'mean'),
        [*np.quantile(values, [0, .05, .5, .95, 1]), values.mean()], strict=True)}


def pool_states(raw_state, length):
    """Nonpadding states are <, the L residues, >, |; use float64 for both means."""
    state = np.asarray(raw_state, dtype=np.float64)
    if (type(length) is not int or length < 1 or state.shape != (length + 3, 480)
            or not np.isfinite(state).all()):
        raise ValueError('Unexpected nonpadding hidden-state dimensions or values.')
    all_mean = state.mean(axis=0)
    residues = state[1:length + 1].mean(axis=0)
    specials = state[[0, length + 1, length + 2]].sum(axis=0)
    if not np.allclose((length + 3) * all_mean, length * residues + specials,
                       rtol=1e-12, atol=1e-10):
        raise ValueError('Pooling identity failed.')
    return all_mean, residues


def rigid_align(values, reference):
    """Least-squares translation + orthogonal transform; reflection allowed, no scaling."""
    from scipy.linalg import orthogonal_procrustes
    values, reference = np.asarray(values, float), np.asarray(reference, float)
    if values.shape != reference.shape or values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('Rigid alignment requires corresponding 2D observations.')
    center, target = values.mean(axis=0), reference.mean(axis=0)
    rotation, _ = orthogonal_procrustes(values - center, reference - target)
    aligned = (values - center) @ rotation + target
    if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-12):
        raise ValueError('Alignment was not orthogonal.')
    return aligned, {'source_center': center.tolist(), 'target_center': target.tolist(),
                     'matrix': rotation.tolist(), 'scale': 1.0,
                     'reflection': bool(np.linalg.det(rotation) < 0),
                     'rms_displacement': float(np.sqrt(np.mean(np.sum((aligned-reference)**2, axis=1))))}


def fixed_density(coordinates, phases, x, y, bandwidth):
    from sklearn.neighbors import KernelDensity
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    phases = np.asarray(phases)
    densities, integrals = [], []
    for phase in PHASES:
        group = coordinates[phases == phase]
        if not len(group):
            raise ValueError('Every phase needs observations.')
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', algorithm='ball_tree',
                            atol=0, rtol=1e-6).fit(group)
        values = np.exp(kde.score_samples(grid)).reshape(len(y), len(x))
        integral = float(np.trapezoid(np.trapezoid(values, x=x, axis=1), x=y))
        if not np.isfinite(integral) or integral <= 0:
            raise ValueError('Invalid density integral.')
        integrals.append(integral)
        densities.append(values / integral)
    return np.asarray(densities), integrals


def load_context(run_dir, projection_dir):
    folder, run, clones, _, hashes = _load_run(Path(run_dir), embeddings=False)
    projection = Path(projection_dir).resolve(strict=True)
    if projection.parent != folder:
        raise ValueError('Projection must belong to the source run.')
    pm = read(projection / 'projection_metadata.json')
    if pm.get('status') != 'completed' or pm['source_sha256'] != hashes:
        raise ValueError('Projection does not match saved source artifacts.')
    ph = pm['artifact_sha256'] | {'projection_metadata.json': digest(projection/'projection_metadata.json')}
    _verify_sources(projection, ph)
    inputs = {p: Path(run['input_paths'][p]) for p in PHASES}
    if {p: digest(v) for p, v in inputs.items()} != run['input_hashes']:
        raise ValueError('Original input differs from the recorded run.')
    if (set(run.get('input_policies', {}).values()) != {'takara-rg-hIGH20181210-v2-ignore-d'}
            or run['embedding']['mode'] != 'official_seqcoding'):
        raise ValueError('Use the current D-ignored, official-seqcoding baseline.')
    for name, expected in pm['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Projection package version differs from baseline.')
    sequences = read(folder/'embedding_sequences.json')
    cached = np.load(folder/'embeddings.npy', allow_pickle=False)
    if cached.shape != (len(sequences), 480) or not np.isfinite(cached).all():
        raise ValueError('Invalid saved embeddings.')
    return folder, run, clones, sequences, cached, hashes, projection, pm, ph, inputs


def verify_unchanged(context):
    folder, run, _, _, _, hashes, projection, _, ph, inputs = context
    _verify_sources(folder, hashes)
    _verify_sources(projection, ph)
    if {p: digest(v) for p, v in inputs.items()} != run['input_hashes']:
        raise ValueError('An original input changed during the comparison.')


def compute_embeddings(context, model_dir, output):
    import torch
    from ablang2.pretrained import format_seq_input
    folder, run, clones, sequences, cached, hashes, projection, pm, ph, inputs = context
    output = validate_output_location(Path(output), inputs)
    if output == folder or folder in output.parents:
        raise ValueError('Comparison must be outside the existing run.')
    output.mkdir(parents=True, exist_ok=False)
    protect_output(output)
    started = perf_counter()
    params = run['parameters']
    encoder = LocalAbLang2(Path(model_dir), batch_size=params['batch_size'],
                          threads=params['threads'], seed=params['seed'])
    for key in ('model','mode','align','fragmented','input_format','pooling','model_parameter_dtype'):
        if encoder.metadata[key] != run['embedding'][key]:
            raise ValueError('Model or input differs from saved baseline.')
    for name in ('ablang2','torch','numpy'):
        if importlib.metadata.version(name) != run['packages'][name]:
            raise ValueError('Embedding package version differs from baseline.')
    # Synthetic strings, not real antibodies: an independent API/dtype/mask check.
    artificial = ['ACDEF', 'ACDEFGHIKLMNPQRSTVWY']
    pairs = [[s, ''] for s in artificial]
    formatted, chain = format_seq_input(pairs, fragmented=False)
    if formatted != ['<' + s + '>|' for s in artificial] or chain != 'H':
        raise ValueError('Unexpected token formatting.')
    with torch.inference_mode():
        states = encoder.model(pairs, mode='rescoding', align=False, fragmented=False,
                               batch_size=params['batch_size'])
        official = encoder.model(pairs, mode='seqcoding', align=False, fragmented=False,
                                 batch_size=params['batch_size'])
    synthetic_means = np.asarray([pool_states(s, len(seq))[0] for s, seq in zip(states, artificial, strict=True)])
    api_error = float(np.max(np.abs(synthetic_means - official)))
    if api_error > 1e-12:
        raise ValueError('Manual mean does not reproduce official seqcoding.')
    all_means, residues = np.empty_like(cached, dtype=np.float64), np.empty_like(cached, dtype=np.float64)
    order = sorted(range(len(sequences)), key=lambda i: (len(sequences[i]), sequences[i]))
    bucket = -1
    raw_dtypes = set()
    with torch.inference_mode():
        for start in range(0, len(order), params['batch_size']):
            ids = order[start:start + params['batch_size']]
            states = encoder.model([[sequences[i], ''] for i in ids], mode='rescoding',
                                   align=False, fragmented=False, batch_size=params['batch_size'])
            for index, state in zip(ids, states, strict=True):
                raw_dtypes.add(str(state.dtype))
                all_means[index], residues[index] = pool_states(state, len(sequences[index]))
            done = min(start + len(ids), len(order)) / len(order)
            if int(done * 20) > bucket:
                bucket = int(done * 20)
                print(f'Paired pooling inference: {done:.0%}', flush=True)
    cache_error = float(np.max(np.abs(all_means - cached)))
    if cache_error > 1e-6:
        raise ValueError('Recomputed baseline differs materially from saved embedding.')
    for name, values in [('all_embeddings.npy', all_means), ('residue_embeddings.npy', residues)]:
        save_array(output/name, values)
    verify_unchanged(context)
    metadata = {'status':'completed', 'source_run':str(folder), 'source_projection':str(projection),
                'source_sha256':hashes, 'projection_sha256':ph, 'input_hashes':run['input_hashes'],
                'encoder':encoder.metadata, 'python':platform.python_version(), 'packages':run['packages'],
                'same_hidden_states_for_both_means':True, 'model_input_unchanged':True,
                'raw_hidden_state_dtypes':sorted(raw_dtypes), 'averaging_dtype':'float64',
                'raw_state_shape_rule':'(CDR_H3_length + 3, 480), no padding',
                'residue_slice':'[1:1+CDR_H3_length]', 'synthetic_api_max_absolute_error':api_error,
                'cached_baseline_max_absolute_error':cache_error,
                'cached_baseline_bitwise_equal':bool(np.array_equal(all_means, cached)),
                'sequence_count':len(sequences), 'all_clone_count':len(clones),
                'batch_order':'length_then_sequence_restored_to_input_order',
                'elapsed_seconds':perf_counter()-started, 'sources_unchanged':True,
                'code':code_provenance(), 'script_sha256':digest(Path(__file__)),
                'artifact_sha256':{n:digest(output/n) for n in ('all_embeddings.npy','residue_embeddings.npy')}}
    write_json(output/'embedding_metadata.json', metadata)
    print('Paired embeddings complete; source and baseline preserved.', flush=True)
    return output


def embedding_metrics(all_vectors, residues, sequences, output):
    from scipy.spatial.distance import pdist, cdist
    from scipy.stats import spearmanr
    # This deterministic length-stratified sample is a diagnostic, not all queries.
    order = np.asarray(sorted(range(len(sequences)), key=lambda i:(len(sequences[i]),i)))
    pair_ids = order[np.unique(np.linspace(0, len(order)-1, min(1024,len(order)), dtype=int))]
    query_ids = order[np.unique(np.linspace(0, len(order)-1, min(512,len(order)), dtype=int))]
    lengths = np.asarray([len(s) for s in sequences])
    norms_a, norms_b = np.linalg.norm(all_vectors,axis=1), np.linalg.norm(residues,axis=1)
    if np.any(norms_a == 0) or np.any(norms_b == 0):
        raise ValueError('Zero vectors cannot be compared by cosine distance.')
    cosine = np.clip(1 - np.sum(all_vectors*residues,axis=1)/(norms_a*norms_b),0,2)
    relative = np.linalg.norm(residues-all_vectors,axis=1)/norms_a
    da, db = pdist(all_vectors[pair_ids],metric='cosine'), pdist(residues[pair_ids],metric='cosine')
    k = min(15, len(sequences)-1)
    neighbors = []
    tie_counts = []
    for vectors in (all_vectors,residues):
        groups=[]; ties=0
        for start in range(0,len(query_ids),32):
            ids=query_ids[start:start+32]
            distances=cdist(vectors[ids], vectors, metric='cosine')
            distances[np.arange(len(ids)),ids]=np.inf
            # Stable sort breaks exact distance ties by stored sequence index.
            nearest=np.argsort(distances,axis=1,kind='stable')[:,:k+1]
            groups.append(nearest[:,:k])
            if k+1<len(sequences):
                d=np.take_along_axis(distances,nearest,axis=1)
                ties+=int(np.count_nonzero(np.isclose(d[:,k-1],d[:,k],rtol=1e-12,atol=1e-14)))
        neighbors.append(np.concatenate(groups))
        tie_counts.append(ties)
    overlap=np.asarray([len(set(a)&set(b))/k for a,b in zip(*neighbors,strict=True)])
    with (output/'embedding_diagnostics.npz').open('xb') as handle:
        np.savez_compressed(handle, pair_ids=pair_ids,query_ids=query_ids,
                            all_neighbors=neighbors[0],residue_neighbors=neighbors[1],overlap=overlap,
                            lengths=lengths,cosine_distance=cosine,relative_l2=relative)
    return {'all_sequence_cosine_distance':summary(cosine), 'all_sequence_relative_l2':summary(relative),
            'pairwise_cosine_distance_spearman':float(spearmanr(da,db).statistic),
            'pair_sample_size':len(pair_ids), 'query_sample_size':len(query_ids),
            'sampling':'deterministic quantiles of length_then_saved_index order; descriptive sample',
            'knn_k':k,'knn_search':'exact cosine against ALL distinct CDR-H3; self excluded',
            'knn_tie_break':'ascending saved sequence index','knn_boundary_near_tie_queries':tie_counts,
            'knn_overlap_fraction':summary(overlap),
            'unit':'distinct_CDR_H3_for_embedding_diagnostics_only',
            'by_length':{str(int(length)):{'n':int((lengths==length).sum()),
                         'median_cosine_distance':float(np.median(cosine[lengths==length]))}
                         for length in np.unique(lengths)}}


def compute_projections(context, output, alternate_seed):
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    from threadpoolctl import threadpool_limits
    folder, run, clones, sequences, cached, hashes, projection, pm, ph, inputs = context
    output = validate_output_location(Path(output).resolve(strict=True), inputs)
    if output == folder or folder in output.parents:
        raise ValueError('Comparison must be outside the existing run.')
    if (output/'comparison_metadata.json').exists():
        raise FileExistsError('Completed comparison exists; choose a new output.')
    em=read(output/'embedding_metadata.json')
    if (em['status']!='completed' or Path(em['source_run'])!=folder
            or em['source_sha256']!=hashes or em['projection_sha256']!=ph):
        raise ValueError('Paired embeddings do not match the requested baseline.')
    _verify_sources(output,em['artifact_sha256'])
    vectors={name:np.load(output/file,allow_pickle=False) for name,file in
             [('all','all_embeddings.npy'),('residue','residue_embeddings.npy')]}
    clone_ids=np.asarray([c['embedding_index'] for c in clones])
    phases=[c['timepoint'] for c in clones]
    seed=pm['parameters']['random_state']
    if type(alternate_seed) is not int or not 0<=alternate_seed<2**32 or alternate_seed==seed:
        raise ValueError('Choose a distinct uint32 alternate seed.')
    seeds=[seed,alternate_seed]
    fits={}; raw={}
    for name in ('all','residue'):
        for current_seed in seeds:
            key=f'{name}_seed{current_seed}'
            settings=pm['parameters'] | {'random_state':current_seed,'transform_seed':current_seed}
            raw_path=output/(key+'_raw.npy')
            fit_record=output/(key+'_fit.json')
            if fit_record.exists():
                saved=read(fit_record)
                if (saved['parameters']!=settings or saved['embedding_sha256']!=digest(output/(name+'_embeddings.npy'))
                        or saved['source_clone_sha256']!=hashes['clones.json']):
                    raise ValueError('Resume fit provenance mismatch.')
                _verify_sources(output,{raw_path.name:saved['raw_sha256']})
                coords=np.load(raw_path,allow_pickle=False)
            else:
                reuse=(name=='all' and current_seed==seed and em['cached_baseline_bitwise_equal'])
                started=perf_counter()
                if reuse:
                    coords=np.load(projection/'coordinates.npy',allow_pickle=False)
                    print('Reusing byte-identical baseline embeddings and verified saved UMAP.',flush=True)
                else:
                    print(f'UMAP starting: {name}, seed {current_seed}',flush=True)
                    expanded=np.asarray(vectors[name][clone_ids],dtype=np.float32)
                    coords=_fit_umap(expanded,settings)
                if coords.shape!=(len(clones),2) or not np.isfinite(coords).all():
                    raise ValueError('UMAP failed to retain every clone observation.')
                save_array(raw_path,coords)
                saved={'parameters':settings,'reused_baseline_coordinates':reuse,
                       'elapsed_seconds':perf_counter()-started,'raw_sha256':digest(raw_path),
                       'source_clone_sha256':hashes['clones.json'],
                       'embedding_sha256':digest(output/(name+'_embeddings.npy'))}
                write_json(fit_record,saved)
            if coords.shape!=(len(clones),2) or not np.isfinite(coords).all():
                raise ValueError('Saved UMAP must contain a finite point for every clone.')
            raw[key]=coords
            fits[key]={'seed':current_seed,'pooling':'all_nonpadding' if name=='all' else 'residue_only',
                       'raw_coordinates_file':raw_path.name,'aligned_coordinates_file':key+'_aligned.npy',
                       'density_file':key+'_density.npz',**saved}
    reference=raw[f'all_seed{seed}']
    aligned={}
    for key,coords in raw.items():
        aligned[key],transform=rigid_align(coords,reference)
        fits[key]['alignment']=transform
        save_array(output/fits[key]['aligned_coordinates_file'],aligned[key])
    bandwidth=float(pm['kde']['bandwidth'])
    stacked=np.concatenate(list(aligned.values()))
    span=np.ptp(stacked,axis=0); margin=np.maximum(3*bandwidth,.05*span)
    lower,upper=stacked.min(axis=0)-margin,stacked.max(axis=0)+margin
    x=np.linspace(lower[0],upper[0],96); y=np.linspace(lower[1],upper[1],96)
    for key,coords in aligned.items():
        print(f'Fixed-bandwidth KDE: {key}',flush=True)
        density,integrals=fixed_density(coords,phases,x,y,bandwidth)
        with (output/fits[key]['density_file']).open('xb') as handle:
            np.savez_compressed(handle,x=x,y=y,densities=density)
        fits[key]['integrals_before_normalization']=dict(zip(PHASES,integrals,strict=True))
    print('Computing embedding-neighborhood and coordinate sensitivity diagnostics.',flush=True)
    with threadpool_limits(limits=run['parameters']['threads']):
        metrics=embedding_metrics(vectors['all'],vectors['residue'],sequences,output)
    # Phase-stratified deterministic clone sample for 2D pairwise distance ranks.
    ids=np.unique(np.concatenate([np.asarray([i for i,c in enumerate(clones) if c['timepoint']==p])[
        np.unique(np.linspace(0,sum(c['timepoint']==p for c in clones)-1,
                              min(350,sum(c['timepoint']==p for c in clones)),dtype=int))] for p in PHASES]))
    pairs=[(f'all_seed{s}',f'residue_seed{s}') for s in seeds]+[
        (f'{n}_seed{seeds[0]}',f'{n}_seed{seeds[1]}') for n in ('all','residue')]
    coordinate_metrics={}
    for a,b in pairs:
        _,transform=rigid_align(raw[b],raw[a])
        coordinate_metrics[a+'__'+b]={'pair_distance_spearman':float(spearmanr(pdist(raw[a][ids]),pdist(raw[b][ids])).statistic),
                                     'rigid_rms_all_clones':transform['rms_displacement']}
    save_array(output/'coordinate_diagnostic_clone_ids.npy',ids)
    verify_unchanged(context)
    metadata={'status':'completed','purpose':'pooling sensitivity, not recovery of original paper settings',
              'source_run':str(folder),'source_projection':str(projection),
              'source_sha256':hashes,'projection_sha256':ph,'input_hashes':run['input_hashes'],
              'input_policies':run['input_policies'],'observation_counts':dict(Counter(phases)),
              'seeds':seeds,'fits':fits,'all_clones_retained':True,'downsampling_for_umap':False,
              'coordinate_alignment':'clone-correspondence rigid transform; translation, rotation/reflection, NO scaling',
              'common_kde':{'bandwidth':bandwidth,'bandwidth_source':'saved current baseline; value fixed for ALL fits',
                            'normalization':'per-timepoint integral one on common finite grid',
                            'grid_size':96,'kernel':'gaussian','rtol':1e-6,'atol':0,'algorithm':'ball_tree',
                            'counts_weights':False,'x_limits':x[[0,-1]].tolist(),'y_limits':y[[0,-1]].tolist()},
              'embedding_metrics':metrics,'coordinate_metrics':coordinate_metrics,
              'coordinate_metric_sample_size':len(ids),'seed_limit':'two seeds are a sensitivity check, not a stability guarantee',
              'production_defaults_changed':False,'candidate_selection_performed':False,
              'paper_coordinate_similarity_quantified':False,'sources_unchanged':True,
              'packages':pm['packages'],'code':code_provenance(),'script_sha256':digest(Path(__file__))}
    metadata['artifact_sha256']={p.name:digest(p) for p in output.iterdir() if p.suffix in ('.npy','.npz','.json')}
    write_json(output/'comparison_metadata.json',metadata)
    print('Pooling comparison completed; all original and baseline artifacts unchanged.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','projection-dir','model-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=('all','embeddings','projections'),default='all')
    parser.add_argument('--alternate-seed',type=int,default=20260920)
    args=parser.parse_args()
    enable_network_guard()
    context=load_context(args.run_dir,args.projection_dir)
    output=args.output_dir
    if args.stage in ('all','embeddings'):
        output=compute_embeddings(context,args.model_dir,output)
    if args.stage in ('all','projections'):
        compute_projections(context,output,args.alternate_seed)


if __name__=='__main__':
    main()
