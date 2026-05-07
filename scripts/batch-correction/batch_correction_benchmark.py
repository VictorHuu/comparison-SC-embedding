import numpy as np, pandas as pd
from scipy import sparse
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import networkx as nx

def to_dense(x): return x.toarray() if sparse.issparse(x) else np.asarray(x)

def cell_embed(X,E,pooling='weighted',topk=50):
    X=to_dense(X).astype(np.float32)
    if pooling=='mean':
        m=(X>0).astype(np.float32); m=m/(m.sum(1,keepdims=True)+1e-8); return m@E
    if pooling=='topk_weighted':
        idx=np.argpartition(X,-min(topk,X.shape[1]),axis=1)[:,-min(topk,X.shape[1]):]
        out=np.zeros((X.shape[0],E.shape[1]),dtype=np.float32)
        for i in range(X.shape[0]):
            w=X[i,idx[i]]; w=w/(w.sum()+1e-8); out[i]=(w[:,None]*E[idx[i]]).sum(0)
        return out
    w=X/(X.sum(1,keepdims=True)+1e-8); return w@E

def linear_residual(C,batch):
    u,inv=np.unique(batch,return_inverse=True)
    B=np.eye(len(u))[inv]
    beta=np.linalg.pinv(B)@C
    return C-B@beta

def graph_conn(C,labels,k=15):
    A=kneighbors_graph(C,n_neighbors=min(k,max(2,C.shape[0]-1)),mode='connectivity',include_self=False)
    G=nx.from_scipy_sparse_array(A)
    vals=[]
    for lab in np.unique(labels):
        idx=np.where(labels==lab)[0]
        if len(idx)<2: continue
        H=G.subgraph(idx)
        cc=max((len(c) for c in nx.connected_components(H)),default=1)
        vals.append(cc/len(idx))
    return float(np.mean(vals)) if vals else np.nan

def run_once(X,labels,batch,E,pooling,correction,seed):
    C=cell_embed(X,E,pooling=pooling)
    if correction=='linear_residual': C=linear_residual(C,batch)
    elif correction=='harmony_optional':
        try:
            import harmonypy as hm
            C=hm.run_harmony(C,pd.DataFrame({'batch':batch}),'batch').Z_corr.T
        except Exception:
            return {'status':'SKIPPED','error_message':'harmonypy unavailable'}
    ncl=len(np.unique(labels))
    cl=KMeans(n_clusters=max(2,ncl),random_state=seed,n_init=10).fit_predict(C)
    nmi=normalized_mutual_info_score(labels,cl); ari=adjusted_rand_score(labels,cl)
    asw_label=silhouette_score(C,labels) if len(np.unique(labels))>1 else np.nan
    asw_batch=1-max(0,silhouette_score(C,batch)) if len(np.unique(batch))>1 else np.nan
    gc=graph_conn(C,labels)
    avgb=np.nanmean([nmi,ari,asw_label]); avgbatch=np.nanmean([asw_batch,gc]); overall=0.6*avgb+0.4*avgbatch
    Xtr,Xte,ytr,yte=train_test_split(C,batch,test_size=0.3,random_state=seed,stratify=batch)
    bacc=LogisticRegression(max_iter=500).fit(Xtr,ytr).score(Xte,yte)
    Xtr,Xte,ytr,yte=train_test_split(C,labels,test_size=0.3,random_state=seed,stratify=labels)
    lacc=LogisticRegression(max_iter=500).fit(Xtr,ytr).score(Xte,yte)
    return dict(NMI_label=nmi,ARI_label=ari,ASW_label=asw_label,AvgBIO=avgb,ASW_batch=asw_batch,GraphConn=gc,AvgBATCH=avgbatch,Overall=overall,batch_predictability_accuracy=bacc,label_predictability_accuracy=lacc,status='OK',error_message='')
