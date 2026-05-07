#!/usr/bin/env python
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import anndata as ad
import matplotlib.pyplot as plt


def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

here=Path(__file__).parent
u=load_local('u',here/'utils_batch_correction.py')
b=load_local('b',here/'batch_correction_benchmark.py')

def pick_layer(adata):
    for k in ['log1p','X_log1p','normalized','counts']:
        if k in adata.layers: return adata.layers[k]
    return adata.X

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-dir',default='.')
    ap.add_argument('--out-dir',default='results/batch-correction')
    ap.add_argument('--datasets',default='auto'); ap.add_argument('--embeddings',default='auto')
    ap.add_argument('--batch-key',default='auto'); ap.add_argument('--label-key',default='auto')
    ap.add_argument('--pooling',default='weighted')
    ap.add_argument('--correction-methods',default='none,linear_residual,harmony_optional')
    ap.add_argument('--n-hvg',type=int,default=2000); ap.add_argument('--seeds',default='0,1,2,3,4')
    ap.add_argument('--max-cells',type=int,default=0); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--strict',action='store_true')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); (out/'plots').mkdir(exist_ok=True); (out/'plots'/'per_dataset_umap').mkdir(parents=True,exist_ok=True)
    embs,dsets,miss=u.discover_project_assets(args.base_dir,args.out_dir)
    edf=pd.DataFrame(embs); ddf=pd.DataFrame(dsets)
    if args.embeddings!='auto': edf=edf[edf.embedding_name.isin(args.embeddings.split(','))]
    if args.datasets!='auto': ddf=ddf[ddf.dataset_name.isin(args.datasets.split(','))]
    ddf=ddf[ddf.status=='OK']
    plan=[]
    for _,d in ddf.iterrows():
        for _,e in edf.iterrows():
            for c in args.correction_methods.split(','):
                for s in [int(x) for x in args.seeds.split(',')]:
                    plan.append(dict(dataset=d.dataset_name,adata_path=d.adata_path,embedding=e.embedding_name,embedding_path=e.embedding_path,gene_list_path=e.gene_list_path,pooling=args.pooling,correction_method=c,seed=s,batch_key=(d.batch_key if args.batch_key=='auto' else args.batch_key),label_key=(d.label_key if args.label_key=='auto' else args.label_key)))
    pd.DataFrame(plan).to_csv(out/'run_plan.csv',index=False)
    if args.dry_run:return
    rows=[]
    for r in plan:
        try:
            adata=ad.read_h5ad(r['adata_path']); X=pick_layer(adata)
            if args.max_cells and adata.n_obs>args.max_cells:
                idx=np.random.default_rng(r['seed']).choice(adata.n_obs,args.max_cells,replace=False); adata=adata[idx].copy(); X=pick_layer(adata)
            labels=adata.obs[r['label_key']].astype(str).to_numpy(); batch=adata.obs[r['batch_key']].astype(str).to_numpy(); genes=adata.var_names.astype(str).to_numpy()
            E=u.load_embedding(r['embedding_path']); gl=u.load_gene_list(r['gene_list_path']) if r['gene_list_path'] else genes[:E.shape[0]].tolist()
            gset={g:i for i,g in enumerate(gl)}; idx=[i for i,g in enumerate(genes) if g in gset]
            if not idx: raise ValueError('no overlap')
            eidx=[gset[genes[i]] for i in idx]; Xo=X[:,idx]; Eo=E[eidx]
            met=b.run_once(Xo,labels,batch,Eo,r['pooling'],r['correction_method'],r['seed'])
            rows.append({**r,'n_cells':adata.n_obs,'n_genes_original':len(genes),'n_genes_overlap':len(idx),'n_hvg':min(args.n_hvg,len(idx)),'n_batches':len(np.unique(batch)),'n_labels':len(np.unique(labels)),**met})
        except Exception as e:
            rows.append({**r,'status':'FAILED','error_message':str(e)})
            if args.strict: raise
    rdf=pd.DataFrame(rows); rdf.to_csv(out/'batch_correction_all_results.csv',index=False)
    ok=rdf[rdf.status=='OK'].copy()
    if len(ok):
        summ=ok.groupby(['dataset','embedding','pooling','correction_method']).agg(['mean','std'])
        summ.columns=['_'.join(c) for c in summ.columns]; summ.reset_index().to_csv(out/'batch_correction_per_dataset_summary.csv',index=False)
        rank=ok.groupby('embedding')[['Overall','AvgBIO','AvgBATCH']].mean().reset_index(); rank['conservative_score']=rank[['AvgBIO','AvgBATCH']].min(1)
        rank.sort_values('Overall',ascending=False).to_csv(out/'batch_correction_rankings.csv',index=False)
        piv=ok.pivot_table(index='embedding',columns='dataset',values='Overall',aggfunc='mean'); plt.figure(figsize=(6,4)); plt.imshow(piv.fillna(0).values); plt.yticks(range(len(piv.index)),piv.index); plt.xticks(range(len(piv.columns)),piv.columns,rotation=45); plt.colorbar(); plt.tight_layout(); plt.savefig(out/'plots'/'overall_heatmap_by_embedding_dataset.png'); plt.close()
        plt.figure(figsize=(5,4)); plt.scatter(ok['AvgBIO'],ok['AvgBATCH']); plt.xlabel('AvgBIO'); plt.ylabel('AvgBATCH'); plt.tight_layout(); plt.savefig(out/'plots'/'avgBIO_vs_avgBATCH_scatter.png'); plt.close()
        ok.groupby('correction_method')['Overall'].mean().plot(kind='bar'); plt.tight_layout(); plt.savefig(out/'plots'/'correction_method_comparison.png'); plt.close()
        rank.set_index('embedding')['Overall'].plot(kind='bar'); plt.tight_layout(); plt.savefig(out/'plots'/'embedding_rank_barplot.png'); plt.close()
    req_present=sorted(set([d for d in ddf['dataset_name'].astype(str).tolist() if any(k in d.lower().replace('_',' ').replace('-',' ') for k in ['pbmc 10k','immune human'])]))
    coverage='NOT_MET' if len(req_present)==0 else ('PARTIAL' if len(req_present)==1 else 'BOTH_PRESENT')
    report=['# Batch-correction report','',f'- scGPT-style required datasets present: {req_present if req_present else []}',f'- scGPT-style coverage status: {coverage}']
    if coverage=='NOT_MET':
        report.append('- Do not claim scGPT-style batch integration coverage (neither PBMC 10K nor Immune Human discovered).')
    elif coverage=='PARTIAL':
        report.append('- Partial scGPT-style coverage only (one required dataset found). Prefer both datasets.')
    (out/'batch_correction_report.md').write_text('\n'.join(report)+'\n')

if __name__=='__main__': main()
