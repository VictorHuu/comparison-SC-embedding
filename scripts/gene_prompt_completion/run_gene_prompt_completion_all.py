#!/usr/bin/env python
import argparse, logging, csv
from pathlib import Path
import numpy as np, pandas as pd
from gene_prompt_completion_benchmark import run_single
from utils import load_gene_list
from gene_prompt_conference_tables import build_conference_tables

DEFAULT_BASE_DIR='/bigdata2/hyt/projects/scbenchmark'
DEFAULT_PERTURB_DATA_DIR=f"{DEFAULT_BASE_DIR}/data/downstreams/perturbation/processed_data"
FIXED_DATASETS=['adamson','dixit','norman']
FIXED_EMBEDDING_KEYS={
    'minus':'module.embedding.weight',
    'baseline':'module.embedding.weight',
    'scGPT_human':'encoder.embedding.weight',
    'v4_bias_rec_best':'embedding.weight',
    'v4_plain_best':'encoder.embedding.weight',
    'v4_type_pe_best':'embedding.weight',
}

def load_embedding(path,key=None):
    p=Path(path); s=p.suffix.lower()
    if s in ['.pt','.pth']:
        import torch
        try:
            obj=torch.load(p,map_location='cpu',weights_only=True)
        except Exception:
            obj=torch.load(p,map_location='cpu',weights_only=False)
        if isinstance(obj,dict):
            keys=[key] if key else []
            keys += ['embedding.weight','encoder.embedding.weight','module.embedding.weight']
            for k in keys:
                if k and k in obj:
                    v=obj[k]; return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
        return np.asarray(obj)
    if s=='.npy': return np.load(p)
    if s=='.npz':
        z=np.load(p)
        for k in z.files:
            if z[k].ndim>=2: return z[k]
    raise ValueError(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-dir',default=DEFAULT_BASE_DIR)
    ap.add_argument('--out-dir',default='results/gene_prompt_completion')
    ap.add_argument('--data-dir',default=DEFAULT_PERTURB_DATA_DIR)
    ap.add_argument('--datasets',default='adamson,dixit,norman')
    ap.add_argument('--embeddings',default='minus,baseline,scGPT_human,v4_bias_rec_best,v4_plain_best,v4_type_pe_best')
    ap.add_argument('--models',default='mean,knn_prompt,ridge_pair,mlp_pair')
    ap.add_argument('--split-modes',default='cell_holdout,gene_holdout')
    ap.add_argument('--prompt-ratios',default='0.05,0.10,0.20')
    ap.add_argument('--target-size',type=int,default=256)
    ap.add_argument('--seeds',default='0,1,2,3,4')
    ap.add_argument('--max-cells',type=int,default=0)
    ap.add_argument('--device',default='auto')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--strict',action='store_true')
    ap.add_argument('--inspect-dataset',default='')
    ap.add_argument('--ragged-policy',default='error',choices=['error','truncate'])
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    log_file=out/'run.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    logger=logging.getLogger('gene_prompt')

    selected_embs=[e.strip() for e in args.embeddings.split(',') if e.strip()]
    selected_datasets=[d.strip() for d in args.datasets.split(',') if d.strip()]
    emb_rows=[]
    for e in selected_embs:
        key=FIXED_EMBEDDING_KEYS.get(e)
        if key is None:
            if args.strict: raise SystemExit(f'Unknown embedding: {e}')
            continue
        emb_rows.append(dict(embedding_name=e,embedding_path=f"{args.base_dir}/save_pretrain/{e}/best_model.pt",embedding_key=key,gene_list_path=f"{args.base_dir}/vocab.json"))
    ds_rows=[]
    for d in selected_datasets:
        raw=Path(d)
        cands=[]
        if raw.suffix.lower() in ['.h5ad','.pt','.parquet'] and raw.exists():
            cands=[raw]
        else:
            cands=[
                Path(args.data_dir)/f'{d}.h5ad',
                Path(args.data_dir)/f'{d}_data.pt',
                Path(args.data_dir)/f'{d}.pt',
                Path(args.data_dir)/f'{d}.parquet',
            ]
        chosen=next((p for p in cands if p.exists()), cands[-1])
        ds_rows.append(dict(dataset_name=raw.stem if raw.suffix else d,adata_path=str(chosen)))
    pd.DataFrame(emb_rows).to_csv(out/'discovered_embeddings.csv',index=False)
    pd.DataFrame(ds_rows).to_csv(out/'discovered_datasets.csv',index=False)

    plan=[]
    for d in ds_rows:
      for e in emb_rows:
        for m in args.models.split(','):
          for sm in args.split_modes.split(','):
            for pr in [float(x) for x in args.prompt_ratios.split(',')]:
              for sd in [int(x) for x in args.seeds.split(',')]:
                plan.append(dict(dataset=d['dataset_name'],adata_path=d['adata_path'],embedding=e['embedding_name'],embedding_path=e['embedding_path'],embedding_key=e['embedding_key'],gene_list_path=e['gene_list_path'],model=m,split_mode=sm,prompt_ratio=pr,seed=sd,device=args.device,target_size=args.target_size))
    pd.DataFrame(plan).to_csv(out/'run_plan.csv',index=False)
    if args.dry_run: return

    import anndata as ad, json, torch
    vocab=list(json.loads(Path(f"{args.base_dir}/vocab.json").read_text()).keys())

    def load_dataset_matrix(path):
        import numpy as np
        p=Path(path)
        if p.suffix=='.parquet':
            import pyarrow.parquet as pq
            from scipy import sparse
            if args.max_cells<=0:
                raise ValueError('Parquet dataset detected. Please set --max-cells to a safe value for conversion (e.g. 50000).')
            pf=pq.ParquetFile(p)
            n_cells=min(args.max_cells,pf.metadata.num_rows)
            n_vocab=len(vocab)
            rows=[]; cols=[]; vals=[]
            seen=0
            for batch in pf.iter_batches(columns=['genes','expressions'], batch_size=4096):
                g_col=batch.column(0); e_col=batch.column(1)
                for i in range(len(g_col)):
                    if seen>=n_cells: break
                    g=np.asarray(g_col[i].as_py(),dtype=np.float64)
                    x=np.asarray(e_col[i].as_py(),dtype=np.float32)
                    m=min(len(g),len(x))
                    if m==0:
                        seen+=1; continue
                    gids=g[:m].astype(np.int64)
                    # common conventions: 0-based or 1-based ids
                    if gids.min()>=1 and gids.max()<=n_vocab: gids=gids-1
                    mask=(gids>=0)&(gids<n_vocab)&np.isfinite(x[:m])
                    rr=np.full(mask.sum(),seen,dtype=np.int64)
                    rows.extend(rr.tolist()); cols.extend(gids[mask].tolist()); vals.extend(x[:m][mask].tolist())
                    seen+=1
                if seen>=n_cells: break
            X=sparse.csr_matrix((np.asarray(vals,dtype=np.float32),(np.asarray(rows),np.asarray(cols))),shape=(seen,n_vocab))
            genes=np.asarray(vocab,dtype=str)
            logger.info(f'Loaded parquet sparse matrix: cells={seen}, genes={n_vocab}, nnz={X.nnz}')
            return X,genes
        if p.suffix=='.h5ad':
            adata=ad.read_h5ad(p)
            X=adata.layers['log1p'] if 'log1p' in adata.layers else adata.X
            genes=adata.var_names.to_numpy()
            return X,genes
        if p.suffix=='.pt':
            obj=torch.load(p,map_location='cpu', weights_only=False)
            if isinstance(obj,dict):
                X=obj.get('X') or obj.get('x') or obj.get('expr') or obj.get('expression') or obj.get('expressions') or obj.get('values')
                genes=obj.get('gene_names') or obj.get('genes') or obj.get('var_names') or obj.get('gene')
                if X is None and 'data' in obj and isinstance(obj['data'],dict):
                    d=obj['data']; X=d.get('X') or d.get('x') or d.get('expressions') or d.get('values'); genes=d.get('gene_names') or d.get('genes') or d.get('var_names')
                if X is None and isinstance(obj.get('expressions'), list): X=obj.get('expressions')
                if X is None and isinstance(obj.get('cells'), list): X=obj.get('cells')
                if X is None:
                    keys=list(obj.keys())
                    raise ValueError(f'Unsupported pt dataset schema: {p}; keys={keys}')
                if isinstance(X, list):
                    X = [np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v) for v in X]
                    if len(X)==0:
                        raise ValueError(f'Empty expression list in {p}')
                    first=np.asarray(X[0])
                    # case A: list of gene-vectors (len(list)==n_genes), each item is per-cell vector
                    if genes is not None and len(X)==len(genes) and first.ndim>=1:
                        lens=[np.asarray(v).reshape(-1).shape[0] for v in X]
                        min_len=min(lens); max_len=max(lens)
                        if min_len!=max_len:
                            uniq=sorted(set(lens))
                            msg=(f'Inconsistent per-gene expression lengths in {p}: min={min_len}, max={max_len}, '
                                 f'unique_sizes={uniq[:8]}{"..." if len(uniq)>8 else ""}. '
                                 f'This usually indicates schema mismatch (not a rectangular cell×gene matrix).')
                            if isinstance(obj.get('base_idx'), list) and isinstance(obj.get('single_ctrl'), list):
                                msg += (' Detected perturbation-style fields base_idx/single_ctrl; this file is likely a '
                                        'gene-indexed perturbation dataset, not a direct cell×gene matrix for prompt completion.')
                            if args.ragged_policy=='error':
                                raise ValueError(msg + ' Use --ragged-policy truncate only for exploratory debugging.')
                            logger.warning(msg + ' Applying truncate policy to min length.')
                        X=np.vstack([np.asarray(v).reshape(-1)[:min_len] for v in X]).T
                    # case B: list of cell-vectors (standard)
                    elif first.ndim == 1:
                        X=np.vstack(X)
                    else:
                        X=np.asarray(X)
                else:
                    X=np.asarray(X.detach().cpu().numpy() if hasattr(X,'detach') else X)
                if X.ndim!=2:
                    raise ValueError(f'Expression matrix must be 2D, got shape={X.shape} from {p}')
                if genes is None: genes=np.array(vocab[:X.shape[1]])
                else: genes=np.asarray(genes).astype(str)
                if len(genes)!=X.shape[1]:
                    raise ValueError(f'Gene count mismatch for {p}: len(genes)={len(genes)} vs X.shape[1]={X.shape[1]}')
                return X,genes
            raise ValueError(f'Unsupported pt object type: {type(obj)}')
        raise ValueError(f'Unsupported dataset suffix: {p.suffix}')

    if args.inspect_dataset:
        ip=Path(args.inspect_dataset)
        if ip.suffix.lower()=='.parquet':
            try:
                import pyarrow.parquet as pq
                pf=pq.ParquetFile(ip)
                print('inspect_dataset type=<parquet>')
                print('num_rows=',pf.metadata.num_rows)
                print('num_row_groups=',pf.num_row_groups)
                print('schema=',pf.schema_arrow)
                for i in range(min(3,pf.num_row_groups)):
                    rg=pf.metadata.row_group(i)
                    print(f' row_group[{i}] rows={rg.num_rows} total_byte_size={rg.total_byte_size}')
            except Exception as e:
                print(f'parquet inspect failed: {e}')
            return
        import torch
        o=torch.load(args.inspect_dataset,map_location='cpu', weights_only=False)
        print('inspect_dataset type=',type(o))
        if isinstance(o,dict):
            print('keys=',list(o.keys()))
            for k,v in o.items():
                shp=getattr(v,'shape',None)
                if isinstance(v,list):
                    shp=f'list(len={len(v)})'
                print(f' - {k}: type={type(v)}, shape={shp}')
            if isinstance(o.get('expressions'), list) and len(o['expressions'])>0:
                import numpy as _np
                lens=[_np.asarray(x).reshape(-1).shape[0] for x in o['expressions'][:200]]
                print(f' expressions_sample_lengths(min/median/max)={min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}')
        return

    rows=[]; gene_rows=[]; manifests=[]
    progress_path=out/'gene_prompt_completion_progress.csv'
    for r in plan:
        try:
            X,genes=load_dataset_matrix(r['adata_path'])
            if args.max_cells and X.shape[0]>args.max_cells: X=X[:args.max_cells]
            E=load_embedding(r['embedding_path'],r['embedding_key'])
            res,gdf,man=run_single(r,X,genes,E,vocab,logging.getLogger('gpc'))
            rows.append({**r,**res})
            with open(progress_path,'a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[-1].keys()))
                if f.tell()==0: w.writeheader()
                w.writerow(rows[-1]); f.flush()
            logger.info(f"OK dataset={r['dataset']} emb={r['embedding']} model={r['model']} split={r['split_mode']} pr={r['prompt_ratio']} seed={r['seed']}")
            gene_rows.append(gdf.assign(dataset=r['dataset'],embedding=r['embedding'],model=r['model'],split_mode=r['split_mode'],prompt_ratio=r['prompt_ratio'],seed=r['seed']))
            manifests.append(dict(dataset=r['dataset'],split_mode=r['split_mode'],prompt_ratio=r['prompt_ratio'],seed=r['seed'],prompt_genes=man['prompt_genes'],target_genes=man['target_genes'],train_target_genes='',heldout_target_genes=''))
        except Exception as e:
            rows.append({**r,'status':'FAILED','error_message':str(e)})
            with open(progress_path,'a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[-1].keys()))
                if f.tell()==0: w.writeheader()
                w.writerow(rows[-1]); f.flush()
            logger.exception(f"FAILED dataset={r['dataset']} emb={r['embedding']} model={r['model']} split={r['split_mode']} pr={r['prompt_ratio']} seed={r['seed']}: {e}")
            if args.strict: raise
    pd.DataFrame(rows).to_csv(out/'gene_prompt_completion_all_results.csv',index=False)
    if gene_rows: pd.concat(gene_rows,ignore_index=True).to_csv(out/'gene_prompt_completion_gene_metrics.csv',index=False)
    pd.DataFrame(manifests).to_csv(out/'prompt_target_manifest.csv',index=False)
    build_conference_tables(out/'gene_prompt_completion_all_results.csv', out)
    (out/'gene_prompt_completion_report.md').write_text('# Gene Prompt Completion Report\n\nSee `gene_prompt_completion_conference_tables.md` for compact conference-style aggregation tables.\n')

if __name__=='__main__': main()
