from pathlib import Path
import os
import re
import json
import pickle
import numpy as np
import pandas as pd


SCBENCH_BASE = os.environ.get("SCBENCH_BASE", "/bigdata2/hyt/projects/scbenchmark")
VOCAB_PATH = f"{SCBENCH_BASE}/vocab.json"

REQUIRED_SCGPT_DATASETS = ["PBMC 10K", "Immune Human"]

BATCH_KEYS = [
    "batch",
    "batch_id",
    "batch_key",
    "donor",
    "sample",
    "sample_id",
    "technology",
    "study",
    "orig.ident",
]

LABEL_KEYS = [
    "cell_type",
    "celltype",
    "cell_type_label",
    "CellType",
    "annotation",
    "label",
    "labels",
    "cell_label",
    "major_celltype",
    "fine_celltype",
]

FIXED_EMBEDDINGS = {
    "minus": {
        "path": f"{SCBENCH_BASE}/save_pretrain/minus/best_model.pt",
        "key": "module.embedding.weight",
    },
    "baseline": {
        "path": f"{SCBENCH_BASE}/save_pretrain/baseline/best_model.pt",
        "key": "module.embedding.weight",
    },
    "scGPT_human": {
        "path": f"{SCBENCH_BASE}/save_pretrain/scGPT_human/best_model.pt",
        "key": "encoder.embedding.weight",
    },
    "v4_bias_rec_best": {
        "path": f"{SCBENCH_BASE}/save_pretrain/v4_bias_rec_best/best_model.pt",
        "key": "embedding.weight",
    },
    "v4_plain_best": {
        "path": f"{SCBENCH_BASE}/save_pretrain/v4_plain_best/best_model.pt",
        "key": "encoder.embedding.weight",
    },
    "v4_type_pe_best": {
        "path": f"{SCBENCH_BASE}/save_pretrain/v4_type_pe_best/best_model.pt",
        "key": "embedding.weight",
    },
}


def _norm_col(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _find_key(cols, candidates):
    m = {_norm_col(c): c for c in cols}
    for c in candidates:
        k = _norm_col(c)
        if k in m:
            return m[k]
    return None


def _get_nested_key(obj, key):
    if key in obj:
        return obj[key]

    cur = obj
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(key)
    return cur


def _as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_embedding_with_key(path, key=None):
    p = Path(path)
    s = p.suffix.lower()

    if s == ".npy":
        arr = np.load(p)
        return arr, key or ""

    if s == ".npz":
        z = np.load(p)
        if key and key in z.files:
            return z[key], key
        for k in z.files:
            if z[k].ndim == 2:
                return z[k], k
        raise ValueError(f"No 2D array found in npz: {path}")

    if s in [".pt", ".pth"]:
        import torch

        obj = torch.load(p, map_location="cpu")

        if isinstance(obj, dict) and key:
            try:
                arr = _as_numpy(_get_nested_key(obj, key))
                return arr, key
            except Exception:
                pass

            if key in obj:
                arr = _as_numpy(obj[key])
                return arr, key

        if isinstance(obj, dict):
            preferred = [
                "embedding.weight",
                "encoder.embedding.weight",
                "module.embedding.weight",
                "gene_embedding.weight",
                "gene_encoder.embedding.weight",
            ]
            for k in preferred:
                if k in obj:
                    arr = _as_numpy(obj[k])
                    if arr.ndim == 2:
                        return arr, k

            for k, v in obj.items():
                try:
                    arr = _as_numpy(v)
                    if arr.ndim == 2 and arr.shape[0] >= 50 and arr.shape[1] >= 8:
                        return arr, k
                except Exception:
                    continue

        arr = _as_numpy(obj)
        if arr.ndim == 2:
            return arr, key or ""

        raise ValueError(f"No usable embedding tensor found in {path}")

    if s == ".pkl":
        with open(p, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and key and key in obj:
            arr = _as_numpy(obj[key])
            return arr, key

        if isinstance(obj, np.ndarray):
            return obj, key or ""

        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    arr = _as_numpy(v)
                    if arr.ndim == 2:
                        return arr, k
                except Exception:
                    continue

        raise ValueError(f"No usable 2D array found in pkl: {path}")

    raise ValueError(f"Unsupported embedding format: {path}")


def load_embedding(path, key=None):
    arr, _ = load_embedding_with_key(path, key)
    return arr


def load_gene_list(path):
    p = Path(path)
    s = p.suffix.lower()

    if s == ".json":
        obj = json.loads(p.read_text())
        if isinstance(obj, dict):
            if all(isinstance(v, int) for v in obj.values()):
                return [k for k, _ in sorted(obj.items(), key=lambda kv: kv[1])]
            return list(obj.keys())
        if isinstance(obj, list):
            return [str(x) for x in obj]
        raise ValueError(f"Unsupported vocab json format: {path}")

    if s in [".csv", ".tsv"]:
        sep = "\t" if s == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        for c in ["gene", "gene_name", "symbol", "feature_name", "gene_symbol", "token"]:
            if c in df.columns:
                return df[c].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()

    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def _dataset_name_from_path(p):
    low = str(p).lower()
    if "pbmc" in low and ("10k" in low or "scvi" in low):
        return "PBMC_10K"
    if "immune" in low and "human" in low:
        return "Immune_Human"
    return Path(p).stem


def _is_valid_label_series(s, key):
    if key is None:
        return False
    if "pseudotime" in str(key).lower():
        return False
    try:
        nunique = s.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(s) and nunique > 100:
            return False
    except Exception:
        return False
    return True


def _validate_dataset_keys(adata, batch_key, label_key):
    if not batch_key or not label_key:
        return False, "missing batch or label key"

    if batch_key not in adata.obs.columns or label_key not in adata.obs.columns:
        return False, "keys not found in obs"

    try:
        nb = adata.obs[batch_key].nunique(dropna=True)
        nl = adata.obs[label_key].nunique(dropna=True)
    except Exception as e:
        return False, f"failed to count groups: {e}"

    if nb < 2:
        return False, f"batch_key={batch_key} has <2 groups"

    if nl < 2:
        return False, f"label_key={label_key} has <2 groups"

    if not _is_valid_label_series(adata.obs[label_key], label_key):
        return False, f"label_key={label_key} looks continuous or invalid"

    return True, f"n_batches={nb},n_labels={nl}"


def _candidate_dataset_files(base):
    base = Path(base)
    candidates = []

    search_dirs = [
        base / "data" / "batch-correction",
        base / "datasets",
        base / "data",
        Path(SCBENCH_BASE) / "data" / "batch-correction",
    ]

    explicit = [
        base / "data" / "batch-correction" / "PBMC_10K_scvi_like.h5ad",
        base / "data" / "batch-correction" / "Immune_Human_openproblems.h5ad",
        base / "data" / "batch-correction" / "Immune_Human_openproblems_prepared.h5ad",
        Path(SCBENCH_BASE) / "data" / "batch-correction" / "PBMC_10K_scvi_like.h5ad",
        Path(SCBENCH_BASE) / "data" / "batch-correction" / "Immune_Human_openproblems.h5ad",
        Path(SCBENCH_BASE) / "data" / "batch-correction" / "Immune_Human_openproblems_prepared.h5ad",
    ]

    for p in explicit:
        if p.exists():
            candidates.append(p)

    for d in search_dirs:
        if d.exists():
            for p in d.rglob("*.h5ad"):
                candidates.append(p)

    bad_tokens = [
        "processed/native",
        "scrna-seq",
        "trajectory",
        "pseudotime",
        "perturbation",
        "adamson",
        "dixit",
        "norman",
    ]

    filtered = []
    seen = set()

    for p in candidates:
        name = str(p).lower()
        if any(t in name for t in bad_tokens):
            continue
        if str(p) not in seen:
            seen.add(str(p))
            filtered.append(p)

    return filtered


def discover_project_assets(base_dir, out_dir):
    import anndata as ad

    base = Path(base_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    emb = []
    miss = []

    for name, cfg in FIXED_EMBEDDINGS.items():
        path = Path(cfg["path"])
        key = cfg["key"]

        if not path.exists():
            miss.append(
                dict(
                    asset_type="embedding",
                    asset_name=name,
                    status="MISSING_PATH",
                    notes=str(path),
                )
            )
            continue

        if not Path(VOCAB_PATH).exists():
            miss.append(
                dict(
                    asset_type="gene_list",
                    asset_name="vocab.json",
                    status="MISSING_PATH",
                    notes=VOCAB_PATH,
                )
            )
            continue

        try:
            mat, used_key = load_embedding_with_key(path, key)
            if mat.ndim != 2:
                raise ValueError(f"embedding ndim={mat.ndim}, expected 2")

            emb.append(
                dict(
                    embedding_name=name,
                    embedding_path=str(path),
                    embedding_key=used_key,
                    gene_list_path=VOCAB_PATH,
                    shape=f"{mat.shape[0]}x{mat.shape[1]}",
                    status="OK",
                    notes="fixed_registry",
                )
            )

        except Exception as e:
            miss.append(
                dict(
                    asset_type="embedding",
                    asset_name=name,
                    status="LOAD_ERROR",
                    notes=f"{path}; key={key}; error={e}",
                )
            )

    ds = []

    for p in _candidate_dataset_files(base):
        try:
            a = ad.read_h5ad(p, backed="r")
            cols = list(a.obs.columns)

            b = _find_key(cols, BATCH_KEYS)
            l = _find_key(cols, LABEL_KEYS)

            ok, note = _validate_dataset_keys(a, b, l)
            st = "OK" if ok else "SKIPPED"

            dname = _dataset_name_from_path(p)

            ds.append(
                dict(
                    dataset_name=dname,
                    adata_path=str(p),
                    batch_key=b or "",
                    label_key=l or "",
                    status=st,
                    notes=note,
                )
            )

            if not ok:
                miss.append(
                    dict(
                        asset_type="dataset",
                        asset_name=dname,
                        status="MISSING_OR_INVALID_KEYS",
                        notes=f"path={p}; batch={b}; label={l}; {note}",
                    )
                )

            try:
                a.file.close()
            except Exception:
                pass

        except Exception as e:
            miss.append(
                dict(
                    asset_type="dataset",
                    asset_name=Path(p).stem,
                    status="ERROR",
                    notes=str(e),
                )
            )

    found = {
        d["dataset_name"].lower().replace("_", "").replace(" ", "")
        for d in ds
        if d["status"] == "OK"
    }

    for req in REQUIRED_SCGPT_DATASETS:
        tok = req.lower().replace("_", "").replace(" ", "")
        hit = any(tok in x or x in tok for x in found)
        if not hit:
            miss.append(
                dict(
                    asset_type="dataset",
                    asset_name=req,
                    status="MISSING_REQUIRED_SCGPT_DATASET",
                    notes="required for scGPT-style batch integration coverage",
                )
            )

    pd.DataFrame(emb).to_csv(out / "discovered_embeddings.csv", index=False)
    pd.DataFrame(ds).to_csv(out / "discovered_datasets.csv", index=False)
    pd.DataFrame(miss).to_csv(out / "missing_assets.csv", index=False)

    report = [
        "# Discovery",
        "",
        "Embedding discovery is disabled.",
        "Embeddings are loaded from fixed registry.",
        "",
        f"SCBENCH_BASE: {SCBENCH_BASE}",
        f"VOCAB_PATH: {VOCAB_PATH}",
        f"Embeddings: {len(emb)}",
        f"Datasets: {len(ds)}",
        f"Missing/skipped/errors: {len(miss)}",
        "",
        "Rules:",
        "- Do not scan scRNA-Seq/ExpressionData.csv as embedding.",
        "- Do not use GeneOrdering.csv as embedding vocabulary.",
        "- Do not use pseudotime as biological label.",
        "- Do not use dataset as default batch key.",
        "- Prefer PBMC 10K and Immune Human for scGPT-style batch integration.",
        "",
    ]

    (out / "asset_discovery_report.md").write_text("\n".join(report))

    return emb, ds, miss
