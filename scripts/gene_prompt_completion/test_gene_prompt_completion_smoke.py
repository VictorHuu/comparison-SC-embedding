from pathlib import Path
import subprocess, sys
import numpy as np, pandas as pd, anndata as ad, torch

def test_smoke(tmp_path: Path):
    base=tmp_path/'repo'; base.mkdir()
    (base/'data'/'downstreams'/'perturbation'/'processed_data').mkdir(parents=True)
    (base/'save_pretrain').mkdir()
    (base/'results').mkdir()
    X=np.random.poisson(1.0,(100,50)).astype(np.float32)
    genes=[f'g{i}' for i in range(50)]
    import torch
    torch.save({'expressions':X.tolist(),'genes':genes,'base_idx':list(range(10)),'single_ctrl':[0]*10,'cls_name':['a']*10}, base/'data'/'downstreams'/'perturbation'/'processed_data'/'adamson_data.pt')
    (base/'vocab.json').write_text('{'+','.join([f'"{g}": {i}' for i,g in enumerate(genes)])+'}')
    for n,k in [('baseline','module.embedding.weight'),('minus','module.embedding.weight')]:
        d=base/'save_pretrain'/n; d.mkdir(parents=True,exist_ok=True)
        E=torch.tensor(np.random.normal(size=(50,16)).astype(np.float32))
        torch.save({k:E},d/'best_model.pt')
    script=Path(__file__).with_name('run_gene_prompt_completion_all.py')
    out=base/'results'/'gpc'
    subprocess.check_call([sys.executable,str(script),'--base-dir',str(base),'--data-dir',str(base/'data'/'downstreams'/'perturbation'/'processed_data'),'--out-dir',str(out),'--dry-run','--datasets','adamson','--embeddings','baseline,minus'])
    subprocess.check_call([sys.executable,str(script),'--base-dir',str(base),'--data-dir',str(base/'data'/'downstreams'/'perturbation'/'processed_data'),'--out-dir',str(out),'--models','mean,ridge_pair','--split-modes','cell_holdout','--prompt-ratios','0.1','--seeds','0','--datasets','adamson','--embeddings','baseline,minus'])
    assert (out/'gene_prompt_completion_all_results.csv').exists()
    df=pd.read_csv(out/'gene_prompt_completion_all_results.csv')
    assert (df['status']=='OK').any()
