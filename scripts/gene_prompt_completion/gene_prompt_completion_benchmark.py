import logging, math
from dataclasses import dataclass
from typing import Dict, List
import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _to_dense(x): return x.toarray() if sparse.issparse(x) else np.asarray(x)

def choose_genes(gene_names, hvg_idx, ratio, seed):
    rng=np.random.default_rng(seed)
    n=max(1,int(len(gene_names)*ratio)); perm=rng.permutation(len(gene_names))
    prompt_idx=np.sort(perm[:n]); t_idx=np.array([i for i in range(len(gene_names)) if i not in set(prompt_idx)])
    return prompt_idx,t_idx

def prompt_cell_repr(Xp, Ep):
    w=Xp/(Xp.sum(1,keepdims=True)+1e-8)
    return (w@Ep).astype(np.float32)

def pair_features(C, Eg):
    n,d=C.shape; m, _=Eg.shape
    C2=np.repeat(C,m,axis=0); E2=np.tile(Eg,(n,1))
    num=(C2*E2).sum(1,keepdims=True); den=(np.linalg.norm(C2,axis=1,keepdims=True)*np.linalg.norm(E2,axis=1,keepdims=True)+1e-8)
    cos=num/den
    return np.concatenate([C2,E2,C2*E2,np.abs(C2-E2),cos],axis=1)

def run_single(cfg, X, gene_names, emb_matrix, emb_gene_names, logger:logging.Logger):
    overlap=[g for g in gene_names if g in set(emb_gene_names)]
    g2i={g:i for i,g in enumerate(gene_names)}; e2i={g:i for i,g in enumerate(emb_gene_names)}
    idx=np.array([g2i[g] for g in overlap]); eidx=np.array([e2i[g] for g in overlap])
    X=_to_dense(X[:,idx]).astype(np.float32); E=emb_matrix[eidx].astype(np.float32)
    tr,te=train_test_split(np.arange(X.shape[0]),test_size=0.2,random_state=cfg['seed'])
    tr,va=train_test_split(tr,test_size=0.2,random_state=cfg['seed'])
    pidx,tidx=choose_genes(overlap,None,cfg['prompt_ratio'],cfg['seed'])
    tcap=int(cfg.get('target_size',0) or 0)
    if tcap>0 and len(tidx)>tcap:
        rng=np.random.default_rng(cfg['seed']+17)
        tidx=np.sort(rng.choice(tidx,size=tcap,replace=False))
    Xtr,Xte=X[tr],X[te]
    Xptr,Xpte=Xtr[:,pidx],Xte[:,pidx]; Xtgtr,Xtgte=Xtr[:,tidx],Xte[:,tidx]
    preds={}
    if cfg['model']=='mean': preds=np.repeat(Xtgtr.mean(0,keepdims=True),len(te),axis=0)
    elif cfg['model']=='knn_prompt':
        nn=NearestNeighbors(n_neighbors=min(10,len(tr))).fit(Xptr); ind=nn.kneighbors(Xpte,return_distance=False); preds=Xtgtr[ind].mean(1)
    else:
        Ctr=prompt_cell_repr(Xptr,E[pidx]); Cte=prompt_cell_repr(Xpte,E[pidx]); Eg=E[tidx]
        Ftr=pair_features(Ctr,Eg); ytr=Xtgtr.reshape(-1)
        Fte=pair_features(Cte,Eg)
        sc=StandardScaler().fit(Ftr); Ftr=sc.transform(Ftr); Fte=sc.transform(Fte)
        if cfg['model']=='ridge_pair': mdl=Ridge(alpha=1.0).fit(Ftr,ytr); yhat=mdl.predict(Fte)
        else:
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
                def forward(self,x): return self.net(x)
            dev='cuda' if cfg.get('device','cpu')=='cuda' and torch.cuda.is_available() else 'cpu'
            m=M(Ftr.shape[1]).to(dev); opt=torch.optim.Adam(m.parameters(),1e-3); lossf=nn.MSELoss()
            xt=torch.tensor(Ftr).float().to(dev); yt=torch.tensor(ytr).float().view(-1,1).to(dev)
            best=1e18; bad=0
            for _ in range(30):
                opt.zero_grad(); loss=lossf(m(xt),yt); loss.backward(); opt.step()
                l=loss.item(); bad=bad+1 if l>best-1e-5 else 0; best=min(best,l)
                if bad>=5: break
            with torch.no_grad(): yhat=m(torch.tensor(Fte).float().to(dev)).cpu().numpy().reshape(-1)
        preds=yhat.reshape(len(te),len(tidx))
    ytrue=Xtgte
    flat_t,flat_p=ytrue.reshape(-1),preds.reshape(-1)
    nz=flat_t>0
    res=dict(mse=mean_squared_error(flat_t,flat_p),mae=mean_absolute_error(flat_t,flat_p),r2=r2_score(flat_t,flat_p),
             pearson_all=float(pearsonr(flat_t,flat_p)[0]) if flat_t.std()>0 and flat_p.std()>0 else np.nan,
             spearman_all=float(spearmanr(flat_t,flat_p)[0]),
             nonzero_mse=mean_squared_error(flat_t[nz],flat_p[nz]) if nz.any() else np.nan,
             nonzero_pearson=float(pearsonr(flat_t[nz],flat_p[nz])[0]) if nz.sum()>2 else np.nan,
             status='OK',error_message='')
    return res, pd.DataFrame([dict(gene=overlap[t],mse=mean_squared_error(ytrue[:,j],preds[:,j]),mae=mean_absolute_error(ytrue[:,j],preds[:,j])) for j,t in enumerate(tidx)]), dict(prompt_genes='|'.join(np.array(overlap)[pidx]),target_genes='|'.join(np.array(overlap)[tidx]))
