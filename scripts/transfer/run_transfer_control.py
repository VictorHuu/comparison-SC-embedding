#!/usr/bin/env python3
"""Run controlled GRN transfer experiments focused on coverage confound."""
import os, csv, json, hashlib
from statistics import mean, pstdev

BASE_DIR = '/bigdata2/hyt/projects/scbenchmark'
SCGREAT_DIR = '/bigdata2/hyt/projects/scGREAT'
OUT_DIR = 'transfer'
os.makedirs(OUT_DIR, exist_ok=True)

STRICT_CSV = os.path.join(OUT_DIR, 'strict_common_gene_seed_results.csv')
COVMATCH_CSV = os.path.join(OUT_DIR, 'coverage_matched_native_results.csv')
GAP_CSV = os.path.join(OUT_DIR, 'transfer_gap_summary.csv')
PER_GENE_CSV = os.path.join(OUT_DIR, 'per_gene_analysis.csv')
SCORE_SHIFT_CSV = os.path.join(OUT_DIR, 'score_shift_summary.csv')
REPORT_MD = os.path.join(OUT_DIR, 'report_transfer_control.md')

EMBEDDINGS = {
    'minus': {'path': f'{BASE_DIR}/save_pretrain/minus/best_model.pt', 'key': 'module.embedding.weight'},
    'baseline': {'path': f'{BASE_DIR}/save_pretrain/baseline/best_model.pt', 'key': 'module.embedding.weight'},
    'scGPT_human': {'path': f'{BASE_DIR}/save_pretrain/scGPT_human/best_model.pt', 'key': 'encoder.embedding.weight'},
}
DATASETS = ['hESC500', 'mESC500']
TRANSFER_DIRS = [('hESC500', 'mESC500'), ('mESC500', 'hESC500')]
SEEDS = [0,1,2,3,4]
SUBSAMPLES = list(range(20))


def write_csv(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_fallback(reason):
    write_csv(STRICT_CSV,
              ['scope','train_dataset','test_dataset','embedding','clf','seed','gene_set_type','coverage_train','coverage_test','auroc','auprc'],
              [])
    write_csv(COVMATCH_CSV,
              ['scope','train_dataset','test_dataset','embedding','clf','subsample_id','matched_gene_count','gene_hash','auroc','auprc'],
              [])
    write_csv(GAP_CSV,
              ['train_dataset','test_dataset','embedding','clf',
               'native_mean_auroc','native_mean_auprc','strict_mean_auroc','strict_mean_auprc',
               'coverage_matched_mean_auroc','coverage_matched_mean_auprc',
               'gap_native_strict_auroc','gap_native_strict_auprc',
               'gap_native_covmatch_auroc','gap_native_covmatch_auprc',
               'delta_vs_baseline_native_auroc','delta_vs_baseline_strict_auroc','delta_vs_baseline_covmatch_auroc',
               'delta_vs_baseline_native_auprc','delta_vs_baseline_strict_auprc','delta_vs_baseline_covmatch_auprc'],
              [])
    write_csv(PER_GENE_CSV,
              ['train_dataset','test_dataset','gene','gene_group','degree','pos_edge_ratio','train_node_freq','test_node_freq','is_tf_proxy','n_genes','median_degree'],
              [])
    write_csv(SCORE_SHIFT_CSV,
              ['train_dataset','test_dataset','embedding','clf','mean_pos_score_source','mean_neg_score_source','margin_source',
               'mean_pos_score_target','mean_neg_score_target','margin_target',
               'abs_shift_pos','abs_shift_neg','abs_shift_margin'],
              [])
    with open(REPORT_MD, 'w') as f:
        f.write('# report_transfer_control\n\n')
        f.write('## 实验目的\n围绕 coverage confound 验证 minus transfer gain 的来源。\n\n')
        f.write('## 实验设置\nstrict common repeated seeds + coverage-matched native + gap decomposition。\n\n')
        f.write('## 主表1：strict common-gene repeated results\n见 strict_common_gene_seed_results.csv\n\n')
        f.write('## 主表2：coverage-matched native results\n见 coverage_matched_native_results.csv\n\n')
        f.write('## 主表3：gap decomposition\n见 transfer_gap_summary.csv\n\n')
        f.write('## 核心结论\n当前环境未完成数值计算，证据不足。\n\n')
        f.write('## limitations\n')
        f.write(f'- {reason}\n')


def main():
    if (not os.path.isdir(SCGREAT_DIR)) or (not os.path.exists(f'{BASE_DIR}/vocab.json')):
        write_fallback('missing /root/autodl-tmp/scGREAT or /root/autodl-tmp/scbenchmark')
        return

    try:
        import numpy as np
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score
    except Exception as e:
        write_fallback(f'missing python deps: {e}')
        return

    def load_vocab():
        with open(f'{BASE_DIR}/vocab.json') as f:
            return json.load(f)

    def load_emb(path, key):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if key in ck: return ck[key].detach().cpu().numpy()
        for nk in ['state_dict','model_state_dict','model']:
            if nk in ck and isinstance(ck[nk], dict) and key in ck[nk]:
                return ck[nk][key].detach().cpu().numpy()
        raise KeyError(key)

    def read_target_genes(ds):
        p = os.path.join(SCGREAT_DIR, ds, 'Target.csv')
        genes=[]
        with open(p, newline='') as f:
            r=csv.DictReader(f)
            for row in r: genes.append(row['Gene'])
        return genes

    def read_split(ds, split):
        p = os.path.join(SCGREAT_DIR, ds, f'{split}.csv')
        tf,tg,y=[],[],[]
        with open(p, newline='') as f:
            r=csv.reader(f); next(r)
            for row in r:
                tf.append(int(row[1])); tg.append(int(row[2])); y.append(int(row[3]))
        return np.array(tf), np.array(tg), np.array(y)

    def build_lookup(emb, vocab, genes):
        d=emb.shape[1]
        lk=np.zeros((len(genes),d),dtype=np.float32)
        mapped=0
        for i,g in enumerate(genes):
            if g in vocab:
                lk[i]=emb[vocab[g]]; mapped+=1
        return lk,mapped

    def pair_features(lk,tf,tg):
        a=lk[tf]; b=lk[tg]
        had=a*b
        cos=np.sum(a*b,axis=1,keepdims=True)/((np.linalg.norm(a,axis=1,keepdims=True)+1e-8)*(np.linalg.norm(b,axis=1,keepdims=True)+1e-8))
        l2=np.linalg.norm(a-b,axis=1,keepdims=True)
        return np.concatenate([a,b,had,cos,l2],axis=1)

    def fit_eval(Xtr,ytr,Xte,yte,clf,seed,resample_lr=False):
        if clf=='lr' and resample_lr:
            rng=np.random.default_rng(seed)
            idx=rng.integers(0,len(ytr),size=len(ytr))
            Xtr=Xtr[idx]; ytr=ytr[idx]
        scaler=StandardScaler(); Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
        if clf=='lr':
            m=LogisticRegression(max_iter=1000,n_jobs=1,C=1.0,random_state=seed)
        else:
            m=MLPClassifier(hidden_layer_sizes=(256,128),max_iter=500,early_stopping=True,random_state=seed)
        m.fit(Xtr,ytr)
        s=m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else m.decision_function(Xte)
        return float(roc_auc_score(yte,s)), float(average_precision_score(yte,s)), s

    def map_pairs_to_genes(pairs_tf, pairs_tg, pairs_y, genes, gene_to_local):
        tf2=[]; tg2=[]; y2=[]
        for a,b,l in zip(pairs_tf,pairs_tg,pairs_y):
            ga=genes[a]; gb=genes[b]
            if ga in gene_to_local and gb in gene_to_local:
                tf2.append(gene_to_local[ga]); tg2.append(gene_to_local[gb]); y2.append(l)
        return np.array(tf2), np.array(tg2), np.array(y2)

    vocab=load_vocab()
    emb_map={k:load_emb(v['path'],v['key']) for k,v in EMBEDDINGS.items()}

    ds={}
    for d in DATASETS:
        genes=read_target_genes(d)
        tr_tf,tr_tg,tr_y=read_split(d,'Train_set')
        va_tf,va_tg,va_y=read_split(d,'Validation_set')
        te_tf,te_tg,te_y=read_split(d,'Test_set')
        ds[d]={
            'genes':genes,
            'train_tf':np.concatenate([tr_tf,va_tf]),
            'train_tg':np.concatenate([tr_tg,va_tg]),
            'train_y':np.concatenate([tr_y,va_y]),
            'test_tf':te_tf,'test_tg':te_tg,'test_y':te_y,
        }

    # cache native available genes per direction/model (train/test side separately)
    native_available={}
    for trd,ted in TRANSFER_DIRS:
        key=(trd,ted); native_available[key]={}
        for emb_name, emb in emb_map.items():
            max_idx = emb.shape[0]
            tr = [g for g in ds[trd]['genes'] if g in vocab and vocab[g] < max_idx]
            te = [g for g in ds[ted]['genes'] if g in vocab and vocab[g] < max_idx]
            native_available[key][emb_name] = {'train': sorted(set(tr)), 'test': sorted(set(te))}

    strict_rows=[]
    native_seed_rows=[]
    covmatch_rows=[]
    score_rows=[]

    for trd,ted in TRANSFER_DIRS:
        key=(trd,ted)
        strict_genes=sorted(
            set(native_available[key]['minus']['train']) & set(native_available[key]['minus']['test']) &
            set(native_available[key]['baseline']['train']) & set(native_available[key]['baseline']['test']) &
            set(native_available[key]['scGPT_human']['train']) & set(native_available[key]['scGPT_human']['test'])
        )
        strict_map={g:i for i,g in enumerate(strict_genes)}

        # strict common repeated seeds (LR+MLP)
        for emb_name, emb in emb_map.items():
            lk_strict=np.stack([emb[vocab[g]] for g in strict_genes]) if strict_genes else np.zeros((0,emb.shape[1]))
            tr_tf,tr_tg,tr_y=map_pairs_to_genes(ds[trd]['train_tf'],ds[trd]['train_tg'],ds[trd]['train_y'],ds[trd]['genes'],strict_map)
            te_tf,te_tg,te_y=map_pairs_to_genes(ds[ted]['test_tf'],ds[ted]['test_tg'],ds[ted]['test_y'],ds[ted]['genes'],strict_map)
            if len(tr_y)<20 or len(te_y)<20 or len(strict_genes)==0:
                continue
            Xtr=pair_features(lk_strict,tr_tf,tr_tg); Xte=pair_features(lk_strict,te_tf,te_tg)
            cov=f'{len(strict_genes)}/{len(ds[trd]["genes"])}'; cov2=f'{len(strict_genes)}/{len(ds[ted]["genes"])}'
            for clf in ['lr','mlp']:
                for seed in SEEDS:
                    au,ap,scores=fit_eval(Xtr,tr_y,Xte,te_y,clf,seed,resample_lr=True)
                    strict_rows.append({'scope':'strict_common_gene_seed','train_dataset':trd,'test_dataset':ted,
                                        'embedding':emb_name,'clf':clf,'seed':seed,'gene_set_type':'strict_common',
                                        'coverage_train':cov,'coverage_test':cov2,'auroc':au,'auprc':ap})

        # native repeated seeds for gap summary + score shift
        for emb_name, emb in emb_map.items():
            gset_tr=native_available[key][emb_name]['train']
            gset_te=native_available[key][emb_name]['test']
            if len(gset_tr)==0 or len(gset_te)==0:
                continue
            gmap_tr={g:i for i,g in enumerate(gset_tr)}
            gmap_te={g:i for i,g in enumerate(gset_te)}
            lk_tr=np.stack([emb[vocab[g]] for g in gset_tr])
            lk_te=np.stack([emb[vocab[g]] for g in gset_te])

            tr_tf,tr_tg,tr_y=map_pairs_to_genes(ds[trd]['train_tf'],ds[trd]['train_tg'],ds[trd]['train_y'],ds[trd]['genes'],gmap_tr)
            te_tf,te_tg,te_y=map_pairs_to_genes(ds[ted]['test_tf'],ds[ted]['test_tg'],ds[ted]['test_y'],ds[ted]['genes'],gmap_te)
            src_tf,src_tg,src_y=map_pairs_to_genes(ds[trd]['test_tf'],ds[trd]['test_tg'],ds[trd]['test_y'],ds[trd]['genes'],gmap_tr)
            if len(tr_y)<20 or len(te_y)<20: continue
            Xtr=pair_features(lk_tr,tr_tf,tr_tg)
            Xte=pair_features(lk_te,te_tf,te_tg)
            Xsrc=pair_features(lk_tr,src_tf,src_tg) if len(src_y)>0 else None
            cov=f'{len(gset_tr)}/{len(ds[trd]["genes"])}'; cov2=f'{len(gset_te)}/{len(ds[ted]["genes"])}'
            for clf in ['lr','mlp']:
                for seed in SEEDS:
                    au,ap,sc_t=fit_eval(Xtr,tr_y,Xte,te_y,clf,seed,resample_lr=True)
                    native_seed_rows.append({'scope':'native_seed','train_dataset':trd,'test_dataset':ted,
                                             'embedding':emb_name,'clf':clf,'seed':seed,'gene_set_type':'native',
                                             'coverage_train':cov,'coverage_test':cov2,'auroc':au,'auprc':ap})
                # score shift use seed0
                au0,ap0,sc_t=fit_eval(Xtr,tr_y,Xte,te_y,clf,0,resample_lr=True)
                if Xsrc is not None and len(src_y)>20:
                    _,_,sc_s=fit_eval(Xtr,tr_y,Xsrc,src_y,clf,0,resample_lr=True)
                    pos_s=float(np.mean(sc_s[src_y==1])) if np.any(src_y==1) else 0.0
                    neg_s=float(np.mean(sc_s[src_y==0])) if np.any(src_y==0) else 0.0
                    pos_t=float(np.mean(sc_t[te_y==1])) if np.any(te_y==1) else 0.0
                    neg_t=float(np.mean(sc_t[te_y==0])) if np.any(te_y==0) else 0.0
                    score_rows.append({'train_dataset':trd,'test_dataset':ted,'embedding':emb_name,'clf':clf,
                                       'mean_pos_score_source':pos_s,'mean_neg_score_source':neg_s,'margin_source':pos_s-neg_s,
                                       'mean_pos_score_target':pos_t,'mean_neg_score_target':neg_t,'margin_target':pos_t-neg_t,
                                       'abs_shift_pos':abs(pos_t-pos_s),'abs_shift_neg':abs(neg_t-neg_s),
                                       'abs_shift_margin':abs((pos_t-neg_t)-(pos_s-neg_s))})

        # coverage matched native (20 subsamples, train/test matched separately)
        min_n_tr=min(len(native_available[key]['minus']['train']),
                     len(native_available[key]['baseline']['train']),
                     len(native_available[key]['scGPT_human']['train']))
        min_n_te=min(len(native_available[key]['minus']['test']),
                     len(native_available[key]['baseline']['test']),
                     len(native_available[key]['scGPT_human']['test']))
        if min_n_tr>0 and min_n_te>0:
            for emb_name, emb in emb_map.items():
                avail_tr=native_available[key][emb_name]['train']
                avail_te=native_available[key][emb_name]['test']
                for sid in SUBSAMPLES:
                    rng_tr=np.random.default_rng(1000+sid)
                    rng_te=np.random.default_rng(5000+sid)
                    pick_tr=sorted(rng_tr.choice(avail_tr,size=min_n_tr,replace=False).tolist())
                    pick_te=sorted(rng_te.choice(avail_te,size=min_n_te,replace=False).tolist())
                    htr=hashlib.md5('\n'.join(pick_tr).encode()).hexdigest()[:12]
                    hte=hashlib.md5('\n'.join(pick_te).encode()).hexdigest()[:12]
                    ghash=f'{htr}-{hte}'
                    gmap_tr={g:i for i,g in enumerate(pick_tr)}
                    gmap_te={g:i for i,g in enumerate(pick_te)}
                    lk_tr=np.stack([emb[vocab[g]] for g in pick_tr])
                    lk_te=np.stack([emb[vocab[g]] for g in pick_te])
                    tr_tf,tr_tg,tr_y=map_pairs_to_genes(ds[trd]['train_tf'],ds[trd]['train_tg'],ds[trd]['train_y'],ds[trd]['genes'],gmap_tr)
                    te_tf,te_tg,te_y=map_pairs_to_genes(ds[ted]['test_tf'],ds[ted]['test_tg'],ds[ted]['test_y'],ds[ted]['genes'],gmap_te)
                    if len(tr_y)<20 or len(te_y)<20: continue
                    Xtr=pair_features(lk_tr,tr_tf,tr_tg); Xte=pair_features(lk_te,te_tf,te_tg)
                    for clf in ['lr','mlp']:
                        au,ap,_=fit_eval(Xtr,tr_y,Xte,te_y,clf,sid,resample_lr=True)
                        covmatch_rows.append({'scope':'coverage_matched_native','train_dataset':trd,'test_dataset':ted,
                                              'embedding':emb_name,'clf':clf,'subsample_id':sid,'matched_gene_count':f'{min_n_tr}/{min_n_te}',
                                              'gene_hash':ghash,'auroc':au,'auprc':ap})

    # per-gene analysis
    per_gene=[]
    group_summary=[]
    for trd,ted in TRANSFER_DIRS:
        key=(trd,ted)
        dset=set(native_available[key]['minus']['train']) | set(native_available[key]['minus']['test'])
        bset=set(native_available[key]['baseline']['train']) | set(native_available[key]['baseline']['test'])
        sset=set(native_available[key]['scGPT_human']['train']) | set(native_available[key]['scGPT_human']['test'])
        common=dset & bset & sset
        only_d=dset - (bset | sset)
        only_b=bset - (dset | sset)
        only_s=sset - (dset | bset)

        # graph stats from train+test edges
        for group, genes in [('common', common), ('minus_only', only_d), ('baseline_only', only_b), ('scGPT_only', only_s)]:
            g2deg={g:0 for g in genes}; g2pos={g:0 for g in genes}; g2tot={g:0 for g in genes}
            g2tr={g:0 for g in genes}; g2te={g:0 for g in genes}
            tf_proxy=set()
            for a,b,l in zip(ds[trd]['train_tf'],ds[trd]['train_tg'],ds[trd]['train_y']):
                ga,gb=ds[trd]['genes'][a],ds[trd]['genes'][b]
                if l==1: tf_proxy.add(ga)
                for g in (ga,gb):
                    if g in g2tot:
                        g2tot[g]+=1; g2pos[g]+=int(l==1); g2tr[g]+=1
            for a,b,l in zip(ds[ted]['test_tf'],ds[ted]['test_tg'],ds[ted]['test_y']):
                ga,gb=ds[ted]['genes'][a],ds[ted]['genes'][b]
                for g in (ga,gb):
                    if g in g2tot:
                        g2tot[g]+=1; g2pos[g]+=int(l==1); g2te[g]+=1
            for g in genes:
                per_gene.append({'train_dataset':trd,'test_dataset':ted,'gene':g,'gene_group':group,
                                 'degree':g2tot[g],'pos_edge_ratio':(g2pos[g]/g2tot[g] if g2tot[g]>0 else 0.0),
                                 'train_node_freq':g2tr[g],'test_node_freq':g2te[g],
                                 'is_tf_proxy':1 if g in tf_proxy else 0})
            if len(genes)>0:
                degs=[g2tot[g] for g in genes]
                prs=[(g2pos[g]/g2tot[g] if g2tot[g]>0 else 0.0) for g in genes]
                trf=[g2tr[g] for g in genes]
                tef=[g2te[g] for g in genes]
                tfr=[1 if g in tf_proxy else 0 for g in genes]
                group_summary.append({
                    'train_dataset':trd,'test_dataset':ted,'gene':'__summary__','gene_group':group,
                    'degree':mean(degs) if degs else 0.0,
                    'pos_edge_ratio':mean(prs) if prs else 0.0,
                    'train_node_freq':mean(trf) if trf else 0.0,
                    'test_node_freq':mean(tef) if tef else 0.0,
                    'is_tf_proxy':mean(tfr) if tfr else 0.0,
                    'n_genes':len(genes),
                    'median_degree':float(np.median(degs)) if degs else 0.0,
                })

    # aggregation for gap summary
    def agg_mean(rows, key_fields, metric):
        d={}
        for r in rows:
            k=tuple(r[kf] for kf in key_fields)
            d.setdefault(k,[]).append(r[metric])
        return {k: (mean(v), (pstdev(v) if len(v)>1 else 0.0)) for k,v in d.items()}

    native_m=agg_mean(native_seed_rows,['train_dataset','test_dataset','embedding','clf'],'auroc')
    native_p=agg_mean(native_seed_rows,['train_dataset','test_dataset','embedding','clf'],'auprc')
    strict_m=agg_mean(strict_rows,['train_dataset','test_dataset','embedding','clf'],'auroc')
    strict_p=agg_mean(strict_rows,['train_dataset','test_dataset','embedding','clf'],'auprc')
    cov_m=agg_mean(covmatch_rows,['train_dataset','test_dataset','embedding','clf'],'auroc')
    cov_p=agg_mean(covmatch_rows,['train_dataset','test_dataset','embedding','clf'],'auprc')

    gap_rows=[]
    for trd,ted in TRANSFER_DIRS:
        for clf in ['lr','mlp']:
            # baseline refs
            b_nat=native_m.get((trd,ted,'baseline',clf),(float('nan'),0))[0]
            b_str=strict_m.get((trd,ted,'baseline',clf),(float('nan'),0))[0]
            b_cov=cov_m.get((trd,ted,'baseline',clf),(float('nan'),0))[0]
            bp_nat=native_p.get((trd,ted,'baseline',clf),(float('nan'),0))[0]
            bp_str=strict_p.get((trd,ted,'baseline',clf),(float('nan'),0))[0]
            bp_cov=cov_p.get((trd,ted,'baseline',clf),(float('nan'),0))[0]
            for emb in ['minus','baseline','scGPT_human']:
                na=native_m.get((trd,ted,emb,clf),(float('nan'),0))[0]
                np_=native_p.get((trd,ted,emb,clf),(float('nan'),0))[0]
                sa=strict_m.get((trd,ted,emb,clf),(float('nan'),0))[0]
                sp=strict_p.get((trd,ted,emb,clf),(float('nan'),0))[0]
                ca=cov_m.get((trd,ted,emb,clf),(float('nan'),0))[0]
                cp=cov_p.get((trd,ted,emb,clf),(float('nan'),0))[0]
                gap_rows.append({
                    'train_dataset':trd,'test_dataset':ted,'embedding':emb,'clf':clf,
                    'native_mean_auroc':na,'native_mean_auprc':np_,
                    'strict_mean_auroc':sa,'strict_mean_auprc':sp,
                    'coverage_matched_mean_auroc':ca,'coverage_matched_mean_auprc':cp,
                    'gap_native_strict_auroc':(na-sa if na==na and sa==sa else ''),
                    'gap_native_strict_auprc':(np_-sp if np_==np_ and sp==sp else ''),
                    'gap_native_covmatch_auroc':(na-ca if na==na and ca==ca else ''),
                    'gap_native_covmatch_auprc':(np_-cp if np_==np_ and cp==cp else ''),
                    'delta_vs_baseline_native_auroc':(na-b_nat if na==na and b_nat==b_nat else ''),
                    'delta_vs_baseline_strict_auroc':(sa-b_str if sa==sa and b_str==b_str else ''),
                    'delta_vs_baseline_covmatch_auroc':(ca-b_cov if ca==ca and b_cov==b_cov else ''),
                    'delta_vs_baseline_native_auprc':(np_-bp_nat if np_==np_ and bp_nat==bp_nat else ''),
                    'delta_vs_baseline_strict_auprc':(sp-bp_str if sp==sp and bp_str==bp_str else ''),
                    'delta_vs_baseline_covmatch_auprc':(cp-bp_cov if cp==cp and bp_cov==bp_cov else ''),
                })

    # write outputs
    write_csv(STRICT_CSV,['scope','train_dataset','test_dataset','embedding','clf','seed','gene_set_type','coverage_train','coverage_test','auroc','auprc'],strict_rows)
    write_csv(COVMATCH_CSV,['scope','train_dataset','test_dataset','embedding','clf','subsample_id','matched_gene_count','gene_hash','auroc','auprc'],covmatch_rows)
    write_csv(GAP_CSV,['train_dataset','test_dataset','embedding','clf',
                       'native_mean_auroc','native_mean_auprc','strict_mean_auroc','strict_mean_auprc',
                       'coverage_matched_mean_auroc','coverage_matched_mean_auprc',
                       'gap_native_strict_auroc','gap_native_strict_auprc',
                       'gap_native_covmatch_auroc','gap_native_covmatch_auprc',
                       'delta_vs_baseline_native_auroc','delta_vs_baseline_strict_auroc','delta_vs_baseline_covmatch_auroc',
                       'delta_vs_baseline_native_auprc','delta_vs_baseline_strict_auprc','delta_vs_baseline_covmatch_auprc'],gap_rows)
    # append group summary rows into per_gene file for quick audit
    per_gene_out=[]
    for r in per_gene:
        x=dict(r); x['n_genes']=''; x['median_degree']=''; per_gene_out.append(x)
    per_gene_out.extend(group_summary)
    write_csv(PER_GENE_CSV,['train_dataset','test_dataset','gene','gene_group','degree','pos_edge_ratio','train_node_freq','test_node_freq','is_tf_proxy','n_genes','median_degree'],per_gene_out)
    write_csv(SCORE_SHIFT_CSV,['train_dataset','test_dataset','embedding','clf','mean_pos_score_source','mean_neg_score_source','margin_source',
                               'mean_pos_score_target','mean_neg_score_target','margin_target',
                               'abs_shift_pos','abs_shift_neg','abs_shift_margin'],score_rows)

    # concise report
    with open(REPORT_MD,'w') as f:
        f.write('# report_transfer_control\n\n')
        f.write('## 实验目的\n验证 minus native transfer gain 是否主要由 coverage confound 导致。\n\n')
        f.write('## 实验设置\nstrict common repeated(5 seeds, LR/MLP, LR含bootstrap重采样)、coverage-matched native(20 subsamples)、gap decomposition。\n\n')
        f.write('## 主表1：strict common-gene repeated results\n见 `strict_common_gene_seed_results.csv`。\n\n')
        f.write('## 主表2：coverage-matched native results\n见 `coverage_matched_native_results.csv`。\n\n')
        f.write('## 主表3：native vs strict vs coverage-matched gap decomposition\n见 `transfer_gap_summary.csv`。\n\n')
        f.write('## 核心结论\n请基于上述三表优先判断：native优势是否在strict/covmatch后消失，以及是否仅在对scGPT_human上稳定。\n\n')
        f.write('## limitations\n')
        f.write('- 若 strict/covmatch 子图样本过少，部分组合可能为空；此时结论应标记为证据不足。\n')


if __name__ == '__main__':
    main()
