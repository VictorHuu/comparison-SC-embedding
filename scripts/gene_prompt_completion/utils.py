import ast, csv, json, logging, os, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

EMB_EXTS={'.npy','.npz','.pt','.pth','.pkl','.csv','.tsv','.txt'}
KNOWN_EMBS=['baseline','minus','scGPT_human','scgpt','geneformer','genecompass','v4_bias_rec_best','v4_plain_best','v4_type_pe_best','difference_v3']
KNOWN_DATASETS=['Adamson','Dixit','Norman','Myeloid','Pancreas','hESC','hHep','mDC','mESC','mHSC-E','mHSC-GM','mHSC-L']

def parse_embedding_names_from_text(text:str)->set[str]:
    found=set()
    for k in KNOWN_EMBS:
        if re.search(re.escape(k), text, re.IGNORECASE): found.add(k)
    for m in re.finditer(r"['\"]([A-Za-z0-9_\-]+)['\"]\s*:\s*\{", text):
        found.add(m.group(1))
    return found

def load_gene_list(path:Path)->List[str]:
    if path.suffix.lower() in ['.csv','.tsv']:
        sep='\t' if path.suffix.lower()=='.tsv' else ','
        df=pd.read_csv(path,sep=sep)
        for c in ['gene','gene_name','symbol','feature_name','var_names']:
            if c in df.columns: return df[c].astype(str).tolist()
        return df.iloc[:,0].astype(str).tolist()
    vals=[]
    with open(path) as f:
        for ln in f:
            s=ln.strip()
            if s: vals.append(s.split(',')[0].split('\t')[0])
    return vals

def probe_embedding_shape(path:Path)->Optional[Tuple[int,int]]:
    try:
        suf=path.suffix.lower()
        if suf=='.npy': a=np.load(path,mmap_mode='r'); return tuple(a.shape[:2]) if a.ndim>=2 else None
        if suf=='.npz': z=np.load(path); 
        
        if suf=='.npz':
            for k in z.files:
                a=z[k]
                if getattr(a,'ndim',0)>=2: return tuple(a.shape[:2])
        if suf in ['.csv','.tsv']:
            sep='\t' if suf=='.tsv' else ','
            df=pd.read_csv(path,sep=sep,nrows=5)
            return (max(5,len(df)),max(1,len(df.columns)))
    except Exception:
        return None
    return None

def discover_project_assets(base_dir:str,out_dir:str,logger:logging.Logger):
    base=Path(base_dir); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    likely=['embeddings','emb','data','outputs','results','processed','checkpoints','save_pretrain','gene_embeddings','scRNA-Seq','datasets']
    script_refs=['benchmark_embeddings.py','perturbation_benchmark.py','scripts/perturbation_regression/perturbation_regression_benchmark.py','grn_embedding_only.py','grn_beeline_full.py']
    script_refs += [str(p.relative_to(base)) for p in (base/'scripts'/'transfer_v2').glob('*.py')] if (base/'scripts'/'transfer_v2').exists() else []
    parsed_expected=set(KNOWN_EMBS)
    for rel in script_refs:
        p=base/rel
        if p.exists():
            txt=p.read_text(errors='ignore'); parsed_expected |= parse_embedding_names_from_text(txt)
            for m in re.finditer(r"['\"](/[^'\"]+|[A-Za-z0-9_\-./]+)['\"]", txt):
                token=m.group(1)
                if '/' in token and len(token)<200:
                    d=(base/token).parent if not token.startswith('/') else Path(token).parent
                    likely.append(d.name)
    likely_dirs=[base/d for d in set(likely) if (base/d).exists()]
    files=[]
    for d in likely_dirs:
        for p in d.rglob('*'):
            if p.is_file(): files.append(p)
    emb_files=[p for p in files if p.suffix.lower() in EMB_EXTS]
    gene_candidates=[p for p in files if any(k in p.name.lower() for k in ['vocab','gene','genes','gene_list','id_to_gene','var_names']) and p.suffix.lower() in ['.txt','.csv','.tsv','.json','.pkl']]
    emb_rows=[]; gl_rows=[]; missing=[]
    grouped={}
    for p in emb_files:
        lname=p.name.lower()
        emb_name=None
        for k in parsed_expected:
            if k.lower() in lname or k.lower() in str(p.parent).lower(): emb_name=k; break
        if emb_name is None:
            emb_name=p.stem
        grouped.setdefault(emb_name,[]).append(p)
    for emb,cands in grouped.items():
        scored=[]
        for c in cands:
            sc=0
            if emb.lower() in c.name.lower(): sc+=2
            sh=probe_embedding_shape(c)
            if sh and sh[0]>10 and sh[1]>4: sc+=2
            nearby=[g for g in gene_candidates if g.parent==c.parent]
            if nearby: sc+=1
            sc += min(c.stat().st_mtime/1e9,10)
            scored.append((sc,c,sh,nearby))
        scored.sort(key=lambda x:x[0],reverse=True)
        best=scored[0]
        gene_list=str(best[3][0]) if best[3] else ''
        shape=f"{best[2][0]}x{best[2][1]}" if best[2] else ''
        emb_rows.append(dict(embedding_name=emb,embedding_path=str(best[1]),gene_list_path=gene_list,shape=shape,status='OK',notes=f'candidates={len(cands)}'))
        for s,c,sh,nb in scored:
            gl_rows.append(dict(embedding_name=emb,candidate_path=str(c),score=s,shape='' if not sh else f'{sh[0]}x{sh[1]}',selected=int(c==best[1])))
    for e in parsed_expected:
        if e not in grouped: missing.append(dict(asset_type='embedding',asset_name=e,status='MISSING_PATH',notes='referenced in scripts/results'))
    ds_files=[p for p in files if p.suffix.lower()=='.h5ad']
    ds_rows=[]
    for p in ds_files:
        nm=p.stem
        ds_rows.append(dict(dataset_name=nm,adata_path=str(p),n_cells='',n_genes='',status='OK',notes=''))
    pd.DataFrame(emb_rows).to_csv(out/'discovered_embeddings.csv',index=False)
    pd.DataFrame(ds_rows).to_csv(out/'discovered_datasets.csv',index=False)
    pd.DataFrame(gl_rows).to_csv(out/'discovered_gene_lists.csv',index=False)
    pd.DataFrame(missing).to_csv(out/'missing_assets.csv',index=False)
    with open(out/'asset_discovery_report.md','w') as f:
        f.write(f"# Asset Discovery\n\nEmbeddings: {len(emb_rows)}\nDatasets: {len(ds_rows)}\nMissing: {len(missing)}\n")
    return emb_rows, ds_rows, missing
