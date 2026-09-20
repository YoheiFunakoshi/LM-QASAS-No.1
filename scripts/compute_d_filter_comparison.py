"""Exploratory display comparison with D annotation/function gates omitted.

This never changes production input rules and does not select new candidates.
All source and output repertoire artifacts must remain local.
"""
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
from lmqasas.inputs import load_three_inputs, _CANONICAL_AA
from lmqasas.pipeline import protect_output, validate_output_location, write_json
from lmqasas import takara
from lmqasas.visualization import _load_run, _verify_sources, _fit_umap, _density_grid

PHASES=('Pre','Peak','Post')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def clone_key(clone):
    return tuple(clone[k] for k in ('timepoint','v_gene','j_gene','cdr3','isotype'))


def read_without_d_gates(paths, hashes):
    """Read already schema/summary-validated workbooks; retain all non-D gates."""
    all_clones=[]
    audits={}
    for phase in PHASES:
        path=paths[phase]
        data=path.read_bytes()
        import hashlib
        if hashlib.sha256(data).hexdigest()!=hashes[phase]:
            raise ValueError('Input changed after full workbook validation.')
        with takara._open(data) as book:
            sheet,_=takara._inspect(book)
            groups={}
            accepted=total=0
            excluded=Counter()
            for row_number,cells in enumerate(sheet.iter_rows(min_row=1,min_col=7,max_col=17),1):
                if row_number>takara.MAX_ROWS:
                    raise ValueError('Worksheet exceeds row limit.')
                if all(c.value is None for c in cells):
                    continue
                if any(c.data_type in ('f','e') for c in cells):
                    raise ValueError('Unexpected formula or error.')
                raw=dict(zip(takara.FIELDS,(takara._safe_raw(c.value) for c in cells),strict=True))
                if not takara._integer(raw['count']):
                    raise ValueError('Invalid count.')
                total+=1
                cdr3=raw['CDR3']
                reasons=[]
                if not isinstance(cdr3,str) or not _CANONICAL_AA.fullmatch(cdr3):
                    reasons.append('cdr3_noncanonical')
                if not isinstance(cdr3,str) or len(cdr3)<5:
                    reasons.append('cdr3_too_short')
                if raw['frame']!='in-frame':
                    reasons.append('frame_not_in_frame')
                for field in ('V','J'):
                    if raw[field+'_function']!='F':
                        reasons.append(field.lower()+'_function_not_F')
                vg,jg=takara._calls(raw['V'],'IGHV'),takara._calls(raw['J'],'IGHJ')
                if vg is None:reasons.append('v_annotation_invalid')
                if jg is None:reasons.append('j_annotation_invalid')
                iso=takara._constant(raw['C'])
                if iso is None:reasons.append('cseg_unmapped_or_ambiguous')
                if reasons:
                    excluded.update(reasons)
                    continue
                accepted+=1
                key=(vg,jg,cdr3,iso)
                if key not in groups:
                    groups[key]={'timepoint':phase,'v_gene':'//'.join(vg),'j_gene':'//'.join(jg),
                                 'cdr3':cdr3,'isotype':iso,'source_file':str(path),
                                 'source_sheet':sheet.title,'source_rows':[]}
                groups[key]['source_rows'].append(row_number)
            all_clones.extend(groups.values())
            audits[phase]={'total_rows':total,'accepted_rows':accepted,'clone_count':len(groups),
                           'reason_counts_nonexclusive':dict(excluded)}
        if digest(path)!=hashes[phase]:raise ValueError('Input changed during read.')
    return all_clones,audits


def legacy_baseline_from_bundle(bundle, saved_audit):
    """Reconstruct strict-v1 only for this historical comparison, never production."""
    samples = saved_audit.get('samples', {})
    if (set(samples) != set(PHASES) or any(
            sample.get('input_policy') != 'takara-rg-hIGH20181210-v1'
            for sample in samples.values())):
        raise ValueError('D-filter comparison requires a saved strict-v1 Excel baseline, not a current v2 run.')
    baseline = []
    for clone in bundle.clones:
        rows, records = [], []
        for row, raw in zip(clone['source_rows'], clone['raw_records'], strict=True):
            if raw['D_function'] == 'F' and takara._calls(raw['D'], 'IGHD') is not None:
                rows.append(row)
                records.append(raw)
        if rows:
            baseline.append({**clone, 'source_row': rows[0], 'source_rows': rows, 'raw_records': records})
    # An ignored D row may have created a group before its first strict-v1 row.
    baseline.sort(key=lambda clone: (PHASES.index(clone['timepoint']), clone['source_row']))
    return baseline


def fixed_grid_density(coordinates, phases, x, y, bandwidth):
    from sklearn.neighbors import KernelDensity
    coordinates=np.asarray(coordinates,dtype=np.float64)
    xx,yy=np.meshgrid(x,y)
    grid=np.column_stack((xx.ravel(),yy.ravel()))
    phases=np.asarray(phases)
    out=[]
    for phase in PHASES:
        kde=KernelDensity(bandwidth=bandwidth,kernel='gaussian',algorithm='ball_tree',
                          atol=0,rtol=1e-6).fit(coordinates[phases==phase])
        values=np.exp(kde.score_samples(grid)).reshape(len(y),len(x))
        integral=np.trapezoid(np.trapezoid(values,x=x,axis=1),x=y)
        if not np.isfinite(integral) or integral<=0:raise ValueError('Invalid density integral.')
        out.append(values/integral)
    return np.asarray(out)


def compute(run_dir, projection_dir, model_dir, output_dir):
    started=perf_counter()
    run_dir,run,baseline,_,source_hashes=_load_run(run_dir,embeddings=True)
    projection=Path(projection_dir).resolve(strict=True)
    if projection.parent!=run_dir:raise ValueError('Projection must belong to baseline run.')
    pm=read(projection/'projection_metadata.json')
    if pm['status']!='completed' or pm['source_sha256']!=source_hashes:
        raise ValueError('Baseline projection source mismatch.')
    projection_hashes=pm['artifact_sha256']|{'projection_metadata.json':digest(projection/'projection_metadata.json')}
    if not {'coordinates.npy','density_grid.npz'}<=projection_hashes.keys():
        raise ValueError('Both baseline projection hashes are required.')
    _verify_sources(projection,projection_hashes)
    paths={p:Path(run['input_paths'][p]) for p in PHASES}
    output=validate_output_location(Path(output_dir),paths)
    if output.exists():raise FileExistsError('Choose a new output directory.')
    if run.get('input_format')!='takara_rg_xlsx':raise ValueError('This diagnostic requires supported RG Excel.')
    input_hashes={p:digest(path) for p,path in paths.items()}
    if input_hashes!=run['input_hashes']:raise ValueError('Inputs differ from saved baseline.')
    # This validates every raw row, schema, count and report summary before the alternate read.
    bundle=load_three_inputs(paths,run['subject'])
    audit_hash=run.get('artifact_sha256',{}).get('input_audit.json')
    if not audit_hash:raise ValueError('Baseline input audit hash is required.')
    _verify_sources(run_dir,{'input_audit.json':audit_hash})
    source_hashes={**source_hashes,'input_audit.json':audit_hash}
    legacy=legacy_baseline_from_bundle(bundle,read(run_dir/'input_audit.json'))
    if [(clone_key(c),c['source_rows']) for c in legacy]!=[(clone_key(c),c['source_rows']) for c in baseline]:
        raise ValueError('Historical strict-v1 baseline does not reproduce the saved clones and source rows.')
    variant,audit=read_without_d_gates(paths,input_hashes)
    variant_index={clone_key(c):i for i,c in enumerate(variant)}
    if len(variant_index)!=len(variant):raise ValueError('Duplicate variant clone keys.')
    baseline_ids=np.array([variant_index[clone_key(c)] for c in baseline],dtype=np.int64)
    output.mkdir(parents=True,exist_ok=False)
    protect_output(output)
    def progress(stage, fraction=None):
        msg=stage if fraction is None else f'{stage}: {fraction:.0%}'
        print(msg,flush=True)
        with (output/'progress.log').open('a',encoding='utf-8') as f:f.write(msg+'\n')
    sequences=sorted({c['cdr3'] for c in variant})
    cached_sequences=read(run_dir/'embedding_sequences.json')
    cache_index={s:i for i,s in enumerate(cached_sequences)}
    cached=np.load(run_dir/'embeddings.npy',allow_pickle=False)
    missing=[s for s in sequences if s not in cache_index]
    seq_index={s:i for i,s in enumerate(sequences)}
    for i,c in enumerate(variant):
        c.update(clone_id=i,embedding_index=seq_index[c['cdr3']])
    write_json(output/'variant_clones.json',variant)
    write_json(output/'variant_embedding_sequences.json',sequences)
    np.save(output/'baseline_in_variant_ids.npy',baseline_ids,allow_pickle=False)
    write_json(output/'input_audit.json',{'variant_rule':'omit_D_function_and_annotation_gates_only',
               'baseline_reproduced':True,'all_baseline_clones_in_variant':True,
               'input_hashes':input_hashes,'samples':audit,'counts_weighted':False})
    progress('Validated baseline and D-omitted clone sets')
    encoder=LocalAbLang2(Path(model_dir),batch_size=run['parameters']['batch_size'],
                        threads=run['parameters']['threads'],seed=run['parameters']['seed'])
    for key in ('model','mode','align','fragmented','input_format','pooling','model_parameter_dtype'):
        if encoder.metadata[key]!=run['embedding'][key]:
            raise ValueError('Embedding cache/model definition mismatch.')
    for package in ('ablang2','torch','numpy'):
        if importlib.metadata.version(package)!=run['packages'][package]:
            raise ValueError('Embedding package version differs from baseline.')
    last_bucket=[-1]
    def inference_progress(stage,fraction):
        bucket=int(fraction*20)
        if bucket>last_bucket[0]:
            progress(stage,fraction)
            last_bucket[0]=bucket
    missing_vectors=encoder.encode(missing,progress=inference_progress) if missing else np.empty((0,480))
    missing_index={s:i for i,s in enumerate(missing)}
    vectors=np.asarray([cached[cache_index[s]] if s in cache_index else missing_vectors[missing_index[s]]
                        for s in sequences],dtype=np.float64)
    if vectors.shape!=(len(sequences),480) or not np.isfinite(vectors).all():
        raise ValueError('Invalid combined embedding matrix.')
    if not np.array_equal(vectors[[seq_index[s] for s in cached_sequences]],cached):
        raise ValueError('Baseline cache was not preserved exactly.')
    np.save(output/'variant_embeddings.npy',vectors,allow_pickle=False)
    write_json(output/'embedding_metadata.json',{'encoder':encoder.metadata,'cached_sequences':len(cached_sequences),
               'new_sequences':len(missing),'all_cached_vectors_unchanged':True,'python':platform.python_version()})
    clone_vectors=np.asarray(vectors[[c['embedding_index'] for c in variant]],dtype=np.float32)
    settings=pm['parameters'].copy()
    if not settings['force_approximation_algorithm']:
        raise ValueError('This diagnostic expects approximate baseline UMAP for a comparable larger fit.')
    progress('UMAP: starting new fit with unchanged hyperparameters')
    coords=_fit_umap(clone_vectors,settings)
    if coords.shape!=(len(variant),2) or not np.isfinite(coords).all():raise ValueError('Invalid UMAP.')
    np.save(output/'variant_coordinates.npy',coords,allow_pickle=False)
    progress('KDE: computing background distributions')
    phases=[c['timepoint'] for c in variant]
    x,y,density,kde=_density_grid(coords,phases,progress=None)
    np.savez_compressed(output/'variant_density_grid.npz',x=x,y=y,densities=density)
    fixed=fixed_grid_density(coords[baseline_ids], [c['timepoint'] for c in baseline],x,y,kde['bandwidth'])
    np.savez_compressed(output/'fixed_baseline_density_grid.npz',x=x,y=y,densities=fixed)
    integrals={name:np.trapezoid(np.trapezoid(d,x=x,axis=2),x=y,axis=1).tolist()
               for name,d in [('variant',density),('fixed_baseline',fixed)]}
    if not all(np.allclose(v,1,atol=1e-8) for v in integrals.values()):
        raise ValueError('Density normalization failed.')
    _verify_sources(run_dir,source_hashes)
    _verify_sources(projection,projection_hashes)
    if any(digest(paths[p])!=input_hashes[p] for p in PHASES):raise ValueError('Original changed.')
    metadata={'status':'completed','baseline_run':str(run_dir),'baseline_projection':str(projection),
              'variant_rule':'ignore_D_annotation_and_function_only','paper_equivalence':'unconfirmed',
              'production_defaults_changed':False,'candidate_selection_performed':False,
              'baseline_counts':dict(Counter(c['timepoint'] for c in baseline)),
              'variant_counts':dict(Counter(phases)),'umap_parameters':settings,
              'baseline_kde':pm['kde'],'variant_kde':kde,
              'bandwidth_rule_unchanged':True,'bandwidth_value_changed_due_to_refit':True,
              'fixed_comparison':'baseline and variant subsets on the same variant UMAP; shared KDE bandwidth/grid',
              'density_integrals':integrals,'input_hashes':input_hashes,
              'source_hashes':source_hashes,'projection_hashes':projection_hashes,
              'sources_unchanged':True,'elapsed_seconds':perf_counter()-started,
              'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','umap-learn','numba','pynndescent')},
              'script_sha256':digest(Path(__file__))}
    metadata['artifact_sha256']={p.name:digest(p) for p in output.iterdir() if p.suffix in ('.json','.npy','.npz')}
    write_json(output/'comparison_inputs.json',metadata)
    progress('Completed D-filter display sensitivity; original and baseline artifacts unchanged')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','projection-dir','model-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    enable_network_guard()
    compute(args.run_dir,args.projection_dir,args.model_dir,args.output_dir)


if __name__=='__main__':main()
