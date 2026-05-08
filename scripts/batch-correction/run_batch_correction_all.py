#!/usr/bin/env python
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd


def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def pick_layer(adata):
    for k in ['log1p','X_log1p','normalized','counts']:
        if k in adata.layers: return adata.layers[k]
    return adata.X



HIGHER_IS_BETTER = {
    "Overall",
    "AvgBIO",
    "AvgBATCH",
    "NMI_label",
    "ARI_label",
    "ASW_label",
    "ASW_batch",
    "GraphConn",
}


def _ordered_embeddings(values):
    preferred = ["baseline", "minus", "geneformer", "scgpt", "uce", "scfoundation", "scbert"]
    vals = [str(v) for v in values]
    return [v for v in preferred if v in vals] + sorted(v for v in vals if v not in preferred)


def _fmt_mean_std(mean, std=None, digits=3):
    if pd.isna(mean):
        return "—"
    text = f"{float(mean):.{digits}f}"
    if std is not None and not pd.isna(std):
        text += f"±{float(std):.{digits}f}"
    return text


def _markdown_table(df, index_name="Setting"):
    if df.empty:
        return "_No data available._"
    table = df.copy()
    table.index = table.index.map(lambda x: " / ".join(map(str, x)) if isinstance(x, tuple) else str(x))
    table = table.reset_index().rename(columns={"index": index_name})
    headers = [str(c) for c in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in table.columns) + " |")
    return "\n".join(lines)


def _style_pivot(summary, metric):
    if summary.empty:
        return summary
    embeddings = list(summary.columns.get_level_values(0).unique())
    styled = pd.DataFrame(index=summary.index, columns=embeddings, dtype=object)
    for idx, row in summary.iterrows():
        means = row.xs("mean", level=1).reindex(embeddings).astype(float)
        stds = row.xs("std", level=1).reindex(embeddings).astype(float) if "std" in row.index.get_level_values(1) else pd.Series(np.nan, index=means.index)
        best = means.max() if metric in HIGHER_IS_BETTER else means.min()
        for emb in means.index:
            value = _fmt_mean_std(means[emb], stds.get(emb, np.nan))
            if pd.notna(means[emb]) and np.isclose(means[emb], best, equal_nan=False):
                value = f"**{value}**"
            styled.loc[idx, emb] = value
    return styled


def export_batch_correction_conference_markdown(results_df, output_dir):
    """Export compact conference-style batch-correction tables from all-results rows."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = results_df.copy()
    ok = df[df["status"].astype(str).str.upper().eq("OK")].copy() if "status" in df.columns else df.copy()
    metrics = [m for m in ["Overall", "AvgBIO", "AvgBATCH", "NMI_label", "ARI_label", "ASW_label", "ASW_batch", "GraphConn"] if m in ok.columns]
    for metric in metrics:
        ok[metric] = pd.to_numeric(ok[metric], errors="coerce")

    lines = [
        "# Batch-Correction Conference Tables",
        "",
        "Style: **bold** marks the best embedding within a row. Values are mean±std across seeds; higher is better for all displayed metrics.",
        "",
        "## Data included",
        "",
        f"- Input rows: {len(df)}",
        f"- Successful rows: {len(ok)}",
        f"- Datasets: {', '.join(map(str, sorted(ok['dataset'].dropna().unique()))) if not ok.empty and 'dataset' in ok.columns else 'none'}",
        f"- Correction methods: {', '.join(map(str, sorted(ok['correction_method'].dropna().unique()))) if not ok.empty and 'correction_method' in ok.columns else 'none'}",
        f"- Metrics shown: {', '.join(metrics) if metrics else 'none'}",
        "",
    ]
    if ok.empty or not metrics:
        lines += ["No successful metric rows available.", ""]
    else:
        rank_cols = [m for m in ["Overall", "AvgBIO", "AvgBATCH"] if m in ok.columns]
        if rank_cols:
            rank = ok.groupby("embedding")[rank_cols].mean().reset_index()
            if {"AvgBIO", "AvgBATCH"}.issubset(rank.columns):
                rank["conservative_score"] = rank[["AvgBIO", "AvgBATCH"]].min(axis=1)
            sort_col = "Overall" if "Overall" in rank.columns else rank_cols[0]
            rank = rank.sort_values(sort_col, ascending=False)
            rank.insert(0, "rank", range(1, len(rank) + 1))
            rank = rank.set_index("embedding").round(4)
            lines += ["## Overall embedding ranking", "", _markdown_table(rank, index_name="Embedding"), ""]

        primary = [m for m in ["Overall", "AvgBIO", "AvgBATCH"] if m in metrics]
        for metric in primary:
            summary = ok.pivot_table(index=["dataset", "correction_method"], columns="embedding", values=metric, aggfunc=["mean", "std"])
            if summary.empty:
                continue
            summary = summary.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
            ordered = _ordered_embeddings(summary.columns.get_level_values(0).unique())
            summary = summary.reindex(columns=pd.MultiIndex.from_product([ordered, ["mean", "std"]]))
            styled = _style_pivot(summary, metric)
            lines += [f"## {metric} by dataset and correction method", "", _markdown_table(styled, index_name="dataset / correction_method"), ""]

        if "Overall" in ok.columns:
            best_rows = []
            grouped = ok.groupby(["embedding", "correction_method"])["Overall"].agg(["mean", "std", "count"]).reset_index()
            for emb, g in grouped.groupby("embedding"):
                row = g.sort_values("mean", ascending=False).iloc[0]
                best_rows.append({
                    "embedding": emb,
                    "best_correction_method": row["correction_method"],
                    "Overall": _fmt_mean_std(row["mean"], row["std"]),
                    "n": int(row["count"]),
                })
            if best_rows:
                best = pd.DataFrame(best_rows).set_index("embedding").loc[_ordered_embeddings([r["embedding"] for r in best_rows])]
                lines += ["## Best correction method per embedding", "", _markdown_table(best, index_name="Embedding"), ""]

        aux = [m for m in ["NMI_label", "ARI_label", "ASW_label", "ASW_batch", "GraphConn"] if m in metrics]
        if aux:
            lines += ["## Auxiliary metric rankings", "", "Mean metric values across all successful datasets, correction methods, and seeds.", ""]
            aux_rank = ok.groupby("embedding")[aux].mean().reindex(_ordered_embeddings(ok["embedding"].dropna().unique())).round(4)
            lines += [_markdown_table(aux_rank, index_name="Embedding"), ""]

        lines += [
            "## Interpretation rules",
            "",
            "- Prefer embeddings that jointly improve Overall, AvgBIO, and AvgBATCH rather than a single metric.",
            "- Compare embeddings within the same dataset and correction method row to avoid mixing correction effects with embedding effects.",
            "- Treat the best-correction table as a workflow-selection summary, not as evidence that one correction method is universally optimal.",
            "",
        ]
    md_path = out / "batch_correction_conference_tables.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path


def export_batch_correction_conference_markdown_from_csv(csv_path, output_dir):
    """Load batch-correction csv and export conference-style markdown directly."""
    df = pd.read_csv(csv_path)
    return export_batch_correction_conference_markdown(df, output_dir)

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
    ap.add_argument('--csv-to-md', default=None, help='Convert an existing batch_correction_all_results.csv to conference-style markdown and exit.')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); (out/'plots').mkdir(exist_ok=True); (out/'plots'/'per_dataset_umap').mkdir(parents=True,exist_ok=True)
    if args.csv_to_md:
        export_batch_correction_conference_markdown_from_csv(args.csv_to_md, out)
        return
    import anndata as ad
    import matplotlib.pyplot as plt
    here=Path(__file__).parent
    u=load_local('u',here/'utils_batch_correction.py')
    b=load_local('b',here/'batch_correction_benchmark.py')
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
    export_batch_correction_conference_markdown(rdf, out)
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
