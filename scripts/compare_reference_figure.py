"""Make a local, labeled visual comparison without refitting or aligning UMAP.

The supplied reference image and existing analysis artifacts remain unchanged.
This produces a comparison sheet, not a numerical reproduction assessment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.image import imread

from lmqasas.checkpoint import digest
from lmqasas.pipeline import protect_output, validate_output_location, write_json
from lmqasas.visualization import _load_run, _load_selection, _verify_sources


def create_comparison(run_dir, projection_dir, reference_image, output_dir,
                      selection_dir=None, font_path=None):
    run_dir, run, clones, _, source_hashes = _load_run(run_dir, embeddings=False)
    projection = Path(projection_dir).resolve(strict=True)
    reference = Path(reference_image).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; prior results are preserved.')
    validate_output_location(output, {k: Path(v) for k,v in run['input_paths'].items()})
    if output == reference.parent or reference.parent in output.parents:
        raise ValueError('Keep comparison output outside the reference image directory.')
    if projection.parent != run_dir:
        raise ValueError('Projection must belong to the selected run.')
    meta_path = projection/'projection_metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('status') != 'completed' or meta.get('source_sha256') != source_hashes:
        raise ValueError('Projection source or completion status failed verification.')
    artifacts = meta.get('artifact_sha256', {})
    if not isinstance(artifacts, dict) or not {'coordinates.npy', 'density_grid.npz'} <= artifacts.keys():
        raise ValueError('Both coordinate and density artifact hashes are required.')
    projection_hashes = {**artifacts, 'projection_metadata.json':digest(meta_path)}
    _verify_sources(projection, projection_hashes)
    coordinates = np.load(projection/'coordinates.npy', allow_pickle=False)
    if (coordinates.shape != (len(clones),2) or not np.isfinite(coordinates).all()
            or hashlib.sha256(coordinates.tobytes(order='C')).hexdigest() != meta['coordinates_content_sha256']):
        raise ValueError('Projection coordinate correspondence failed verification.')
    with np.load(projection/'density_grid.npz', allow_pickle=False) as grid:
        x,y,densities = (grid[key] for key in ('x','y','densities'))
    if (densities.shape != (3,len(y),len(x)) or not np.isfinite(densities).all()
            or np.any(densities < 0) or np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0)
            or not np.allclose(np.trapezoid(np.trapezoid(densities,x=x,axis=2),x=y,axis=1),1,atol=1e-8)):
        raise ValueError('Saved density grids failed verification.')
    selection, summary, selected, selection_hashes = _load_selection(run_dir,run,selection_dir)
    reference_hash = digest(reference)
    reference_pixels = imread(reference)
    font = font_manager.FontProperties(fname=str(font_path)) if font_path else None
    output.mkdir(parents=True,exist_ok=False)
    protect_output(output)
    fig=Figure(figsize=(16,12),dpi=180,facecolor='white')
    FigureCanvasAgg(fig)
    def text(xpos,ypos,value,size=12,**kwargs):
        return fig.text(xpos,ypos,value,fontsize=size,fontproperties=font,**kwargs)
    text(.05,.962,'論文 Fig. 1c と現行アプリの比較',21,weight='bold')
    text(.05,.927,'A  論文 Fig. 1c（提供された参照画像）',15,weight='bold')
    text(.05,.902,'赤点：既知の CoV-AbDab 登録配列との類似性（長さ正規化編集距離 ≤ 0.15）',12)
    upper=fig.add_axes([.035,.611,.93,.274])
    upper.imshow(reference_pixels)
    upper.set_axis_off()
    text(.05,.584,'出典：Masuda et al., Frontiers in Immunology (2026), Fig. 1c. DOI: 10.3389/fimmu.2026.1844788',10,color='#45515c')
    text(.05,.541,'B  現行アプリ（保存済みの座標・密度から再作図）',15,weight='bold')
    text(.05,.515,f'橙色の印：Peak から選んだ {summary["returned"]:,} 種類の CDR-H3 と完全一致する clone 観測',12)
    phases=('Pre','Peak','Post')
    levels=np.linspace(0,float(densities.max()),21)
    overlay={}
    for i,phase in enumerate(phases):
        ax=fig.add_axes([.055+i*.287,.153,.25,.323])
        contour=ax.contourf(x,y,densities[i],levels=levels,cmap='viridis',vmin=0,vmax=levels[-1])
        ax.contour(x,y,densities[i],levels=levels[1:-1:2],colors='#394a55',alpha=.35,linewidths=.35)
        ids=[j for j,c in enumerate(clones) if c['timepoint']==phase and c['cdr3'] in selected]
        overlay[phase]=ids
        pts=coordinates[ids]
        ax.scatter(pts[:,0],pts[:,1],s=10,c='#ff941f',edgecolors='#2b1b08',linewidths=.25,alpha=.8)
        ax.set_title(phase,fontsize=13,weight='bold',pad=9)
        ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),xlabel='Shared UMAP 1')
        if i==0:ax.set_ylabel('Shared UMAP 2')
        ax.tick_params(labelsize=9)
        ax.set_aspect('equal',adjustable='box')
    cbar=fig.colorbar(contour,cax=fig.add_axes([.915,.153,.014,.323]))
    cbar.set_label('Normalized density',fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    text(.05,.09,'上下で座標系・密度尺度・強調対象が異なります。点の位置や色の数値は直接比較できません。',12,weight='bold')
    text(.05,.061,'この比較だけで、論文の再現成功や抗原特異性の一致を判断することはできません。',11)
    text(.05,.033,'同一条件での検証には、元の前処理・AbLang2・UMAP/KDE設定と、使用した参照DBまたは対応ラベルが必要です。',10,color='#45515c')
    fig.savefig(output/'fig1c_comparison.png',dpi=180,facecolor='white')
    fig.clear()
    _verify_sources(run_dir,source_hashes)
    _verify_sources(projection,projection_hashes)
    _verify_sources(selection,selection_hashes)
    if digest(reference) != reference_hash:
        raise ValueError('Reference image changed during rendering.')
    write_json(output/'comparison_metadata.json',{
        'status':'completed','source_run':str(run_dir),'projection':str(projection),
        'selection':str(selection),'reference_image':str(reference),
        'reference_sha256':reference_hash,'source_sha256':source_hashes,
        'projection_sha256':projection_hashes,'selection_sha256':selection_hashes,
        'selected_clone_ids':overlay,'selection_summary':summary,
        'umap_refitted':False,'coordinate_alignment_attempted':False,
        'cov_abdab_matching_performed':False,'paper_reproduction_confirmed':False,
        'source_files_unchanged':True,'outputs_private':True,
        'renderer_sha256':digest(Path(__file__)),
        'artifact_sha256':{'fig1c_comparison.png':digest(output/'fig1c_comparison.png')}})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','projection-dir','reference-image','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--selection-dir',type=Path)
    parser.add_argument('--font-path',type=Path,default=Path('C:/Windows/Fonts/meiryo.ttc'))
    args=parser.parse_args()
    try:
        create_comparison(args.run_dir,args.projection_dir,args.reference_image,args.output_dir,
                          args.selection_dir,args.font_path)
    except Exception as exc:
        print(f'Comparison stopped ({type(exc).__name__}); no source-writing operations were performed.',file=sys.stderr)
        return 1
    print('Comparison figure saved; no analysis or reference files were modified.')
    return 0


if __name__=='__main__':raise SystemExit(main())
