from pathlib import Path
import subprocess, sys, numpy as np, pandas as pd, anndata as ad

def test_smoke(tmp_path: Path):
    base=tmp_path/'repo'; (base/'data').mkdir(parents=True); (base/'embeddings').mkdir(); (base/'results').mkdir()
    n,m=200,80
    rng=np.random.default_rng(0)
    batch=np.array(['b0']*(n//2)+['b1']*(n-n//2)); label=np.array([f'ct{i%3}' for i in range(n)])
    X=rng.poisson(1.0,(n,m)).astype(np.float32); X[batch=='b1',:10]+=1
    adata=ad.AnnData(X=X); adata.obs['batch']=batch; adata.obs['cell_type']=label; adata.var_names=[f'g{i}' for i in range(m)]
    adata.write_h5ad(base/'data'/'toy.h5ad')
    for e in ['baseline','minus']:
        E=rng.normal(size=(m,16)).astype(np.float32); np.save(base/'embeddings'/f'{e}.npy',E); pd.Series(adata.var_names).to_csv(base/'embeddings'/f'{e}_genes.txt',index=False,header=False)
    script=Path(__file__).with_name('run_batch_correction_all.py'); out=base/'results'/'batch-correction'
    subprocess.check_call([sys.executable,str(script),'--base-dir',str(base),'--out-dir',str(out),'--correction-methods','none,linear_residual','--seeds','0','--pooling','weighted'])
    assert (out/'batch_correction_all_results.csv').exists()
    assert (out/'discovered_embeddings.csv').exists()
    assert (out/'discovered_datasets.csv').exists()
    assert (out/'batch_correction_report.md').exists()
    df=pd.read_csv(out/'batch_correction_all_results.csv'); assert (df['status']=='OK').any()
