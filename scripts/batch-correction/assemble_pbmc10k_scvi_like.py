#!/usr/bin/env python3
import os
import sys
import tarfile
import pickle
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


RAW_DIR = Path("data/batch-correction/pbmc10k_raw")
OUT_PATH = Path("data/batch-correction/PBMC_10K_scvi_like.h5ad")

PBMC8K_TAR = RAW_DIR / "pbmc8k_filtered_gene_bc_matrices.tar.gz"
PBMC4K_TAR = RAW_DIR / "pbmc4k_filtered_gene_bc_matrices.tar.gz"
GENE_INFO = RAW_DIR / "gene_info_pbmc.csv"
METADATA = RAW_DIR / "pbmc_metadata.pickle"


def check_inputs():
    required = [PBMC8K_TAR, PBMC4K_TAR, GENE_INFO, METADATA]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def install_legacy_pandas_pickle_shim():
    mod_name = "pandas.core.indexes.numeric"
    if mod_name in sys.modules:
        return
    mod = types.ModuleType(mod_name)
    mod.Int64Index = pd.Index
    mod.UInt64Index = pd.Index
    mod.Float64Index = pd.Index
    mod.NumericIndex = pd.Index
    sys.modules[mod_name] = mod


def load_metadata_pickle(path: Path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        if "pandas.core.indexes.numeric" not in str(e):
            raise
        install_legacy_pandas_pickle_shim()
        with open(path, "rb") as f:
            return pickle.load(f)


def extract_tar(tar_path: Path, out_dir: Path):
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"[extract] skip existing: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tar_path} -> {out_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)
    marker.write_text("ok\n")


def find_10x_dir(root: Path) -> Path:
    for dirpath, _, filenames in os.walk(root):
        files = set(filenames)
        if (
            ("matrix.mtx" in files or "matrix.mtx.gz" in files)
            and ("barcodes.tsv" in files or "barcodes.tsv.gz" in files)
            and (
                "genes.tsv" in files
                or "genes.tsv.gz" in files
                or "features.tsv" in files
                or "features.tsv.gz" in files
            )
        ):
            return Path(dirpath)
    raise FileNotFoundError(f"Cannot find 10X matrix directory under {root}")


def read_one_10x(name: str, tar_path: Path, batch_code: str) -> sc.AnnData:
    extract_dir = RAW_DIR / f"{name}_extracted"
    extract_tar(tar_path, extract_dir)
    mtx_dir = find_10x_dir(extract_dir)
    print(f"[read] {name}: {mtx_dir}")

    x = sc.read_10x_mtx(str(mtx_dir), var_names="gene_ids", cache=False)

    raw = pd.Series(x.obs_names.astype(str), index=x.obs_names)
    no_suffix = raw.str.replace(r"-1$", "", regex=True)

    x.obs["barcode_orig"] = raw.values
    x.obs["batch"] = str(batch_code)

    # Candidate formats seen across old scVI / 10X conventions.
    x.obs["bc_raw"] = raw.values
    x.obs["bc_raw_plus_batch"] = (raw + str(batch_code)).values              # AAAC...-10
    x.obs["bc_raw_dash_batch"] = (raw + "-" + str(batch_code)).values        # AAAC...-1-0
    x.obs["bc_no_suffix_plus_batch"] = (no_suffix + str(batch_code)).values  # AAAC...0
    x.obs["bc_no_suffix_dash_batch"] = (no_suffix + "-" + str(batch_code)).values
    x.obs["bc_no_suffix_under_batch"] = (no_suffix + "_" + str(batch_code)).values

    print(f"[read] {name} shape: {x.shape}")
    return x


def metadata_barcode_vectors(meta: dict, n: int):
    out = []

    def add(name, values):
        if values is None:
            return
        arr = np.asarray(values).astype(str).reshape(-1)
        if len(arr) == n:
            out.append((name, arr))

    for key in ["barcodes", "design", "raw_qc", "qc_pc", "normalized_qc"]:
        if key not in meta:
            continue
        obj = meta[key]

        if isinstance(obj, pd.DataFrame):
            add(f"{key}.index", obj.index.astype(str).to_numpy())
            for col in obj.columns:
                if obj[col].dtype == object or str(obj[col].dtype).startswith("category"):
                    add(f"{key}.{col}", obj[col].astype(str).to_numpy())
        elif isinstance(obj, pd.Series):
            add(f"{key}.index", obj.index.astype(str).to_numpy())
            add(f"{key}.values", obj.astype(str).to_numpy())
        else:
            arr = np.asarray(obj)
            if arr.ndim == 1:
                add(f"{key}.values", arr.astype(str))
            elif arr.ndim == 2:
                for j in range(arr.shape[1]):
                    add(f"{key}.col{j}", arr[:, j].astype(str))

    # Deduplicate identical vectors.
    seen = set()
    unique = []
    for name, arr in out:
        sig = tuple(arr[:20]) + (len(arr),)
        if sig not in seen:
            seen.add(sig)
            unique.append((name, arr))
    return unique


def build_candidate_lookup(adata: sc.AnnData):
    cols = [
        "bc_raw",
        "bc_raw_plus_batch",
        "bc_raw_dash_batch",
        "bc_no_suffix_plus_batch",
        "bc_no_suffix_dash_batch",
        "bc_no_suffix_under_batch",
    ]

    lookup = {}
    duplicated = set()

    for i in range(adata.n_obs):
        row = adata.obs.iloc[i]
        for col in cols:
            key = str(row[col])
            if key in lookup:
                duplicated.add(key)
            else:
                lookup[key] = i

    for key in duplicated:
        lookup.pop(key, None)

    print(f"[lookup] usable keys: {len(lookup)}")
    print(f"[lookup] duplicate keys removed: {len(duplicated)}")
    return lookup


def choose_best_barcode_vector(meta, adata, clusters):
    lookup = build_candidate_lookup(adata)
    vectors = metadata_barcode_vectors(meta, len(clusters))

    best_name, best_arr, best_matches = None, None, []

    print("[metadata barcode candidates]")
    for name, arr in vectors:
        matches = []
        used_obs = set()
        for j, bc in enumerate(arr):
            idx = lookup.get(str(bc))
            if idx is not None and idx not in used_obs:
                matches.append((idx, j))
                used_obs.add(idx)
        print(f"  {name}: {len(matches)} matches")
        if len(matches) > len(best_matches):
            best_name, best_arr, best_matches = name, arr, matches

    return best_name, best_arr, best_matches


def infer_batch_from_design(meta, n):
    if "design" not in meta:
        return None

    d = meta["design"]

    if isinstance(d, pd.DataFrame):
        if len(d) != n:
            return None

        # First try a categorical / low-cardinality column.
        for col in d.columns:
            s = d[col]
            vals = s.astype(str).to_numpy()
            uniq = pd.unique(vals)
            if 1 < len(uniq) <= 4:
                codes = pd.factorize(vals)[0]
                return codes

        # Then try one-hot numeric matrix.
        num = d.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            arr = num.to_numpy()
            return np.argmax(arr[:, :2], axis=1)

    elif isinstance(d, pd.Series):
        if len(d) != n:
            return None
        vals = d.astype(str).to_numpy()
        if len(pd.unique(vals)) > 1:
            return pd.factorize(vals)[0]

    else:
        arr = np.asarray(d)
        if arr.ndim == 1 and len(arr) == n:
            return pd.factorize(arr.astype(str))[0]
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] >= 2:
            return np.argmax(arr[:, :2], axis=1)

    return None


def fallback_match_with_design(meta, adata, clusters):
    vectors = metadata_barcode_vectors(meta, len(clusters))
    raw_vec = None
    raw_name = None
    for name, arr in vectors:
        # Prefer raw-looking barcode vector.
        if np.mean([str(x).endswith("-1") for x in arr[:100]]) > 0.8:
            raw_name, raw_vec = name, arr
            break

    if raw_vec is None:
        return None, None, []

    batch_codes = infer_batch_from_design(meta, len(clusters))
    if batch_codes is None:
        return None, None, []

    # Build per-batch raw barcode lookup.
    per_batch = {"0": {}, "1": {}}
    for i in range(adata.n_obs):
        b = str(adata.obs.iloc[i]["batch"])
        bc = str(adata.obs.iloc[i]["bc_raw"])
        if b in per_batch:
            per_batch[b][bc] = i

    def try_mapping(reverse=False):
        matches = []
        used = set()
        for j, bc in enumerate(raw_vec):
            code = int(batch_codes[j])
            if reverse:
                code = 1 - code
            b = str(code)
            idx = per_batch.get(b, {}).get(str(bc))
            if idx is not None and idx not in used:
                matches.append((idx, j))
                used.add(idx)
        return matches

    m1 = try_mapping(reverse=False)
    m2 = try_mapping(reverse=True)

    if len(m2) > len(m1):
        return f"{raw_name}+design(reversed)", raw_vec, m2
    return f"{raw_name}+design", raw_vec, m1


def labels_from_clusters(clusters, list_clusters):
    clusters = np.asarray(clusters).reshape(-1)
    try:
        c = clusters.astype(int)
        labels = c.astype(str)
        str_labels = np.array([
            str(list_clusters[i]) if 0 <= i < len(list_clusters) else f"label_{i}"
            for i in c
        ])
        return labels, str_labels
    except Exception:
        str_labels = clusters.astype(str)
        labels = pd.factorize(str_labels)[0].astype(str)
        return labels, str_labels


def attach_metadata(adata):
    print(f"[metadata] loading: {METADATA}")
    meta = load_metadata_pickle(METADATA)

    print("[metadata] keys:", list(meta.keys()))
    clusters = np.asarray(meta["clusters"]).reshape(-1)
    list_clusters = [str(x) for x in list(meta["list_clusters"])]

    print(f"[metadata] rows: {len(clusters)}")
    print(f"[metadata] label names: {len(list_clusters)}")

    best_name, best_arr, matches = choose_best_barcode_vector(meta, adata, clusters)

    if len(matches) < 10000:
        print(f"[metadata] best direct match only {len(matches)}; trying design-based fallback")
        fb_name, fb_arr, fb_matches = fallback_match_with_design(meta, adata, clusters)
        if len(fb_matches) > len(matches):
            best_name, best_arr, matches = fb_name, fb_arr, fb_matches

    print(f"[metadata] selected barcode source: {best_name}")
    print(f"[metadata] selected matches: {len(matches)}")

    if len(matches) < 10000:
        print("[debug] first 20 selected metadata barcodes:")
        if best_arr is not None:
            print(best_arr[:20])
        print("[debug] first 10 adata candidates:")
        print(adata.obs[[
            "batch",
            "bc_raw",
            "bc_raw_plus_batch",
            "bc_raw_dash_batch",
            "bc_no_suffix_plus_batch",
            "bc_no_suffix_dash_batch",
        ]].head(10))
        raise RuntimeError("Still too few metadata matches. Need inspect metadata['barcodes'] and metadata['design'].")

    obs_idx = [x[0] for x in matches]
    meta_idx = [x[1] for x in matches]

    labels, str_labels = labels_from_clusters(clusters, list_clusters)

    adata = adata[obs_idx, :].copy()
    adata.obs["labels"] = labels[meta_idx].astype(str)
    adata.obs["str_labels"] = str_labels[meta_idx].astype(str)
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    print("[after metadata]", adata)
    print("[batch counts]")
    print(adata.obs["batch"].value_counts())
    print("[celltype counts]")
    print(adata.obs["celltype"].value_counts())

    return adata


def filter_genes(adata):
    gene_info = pd.read_csv(GENE_INFO)
    print("[gene_info columns]", list(gene_info.columns))

    if "ENSG" not in gene_info.columns:
        raise ValueError("gene_info_pbmc.csv must contain column ENSG")

    target = gene_info["ENSG"].astype(str).tolist()
    present = set(adata.var_names.astype(str))
    keep = [g for g in target if g in present]

    print(f"[gene filter] target={len(target)}, overlap={len(keep)}")

    if len(keep) < 3000:
        raise RuntimeError(f"Too few genes retained: {len(keep)}; expected near 3346")

    adata = adata[:, keep].copy()
    adata.var["gene_ids"] = adata.var_names.astype(str)

    if "gene_symbols" in adata.var.columns:
        adata.var["gene_name"] = adata.var["gene_symbols"].astype(str).values
        adata.var_names = adata.var["gene_name"].astype(str)
        adata.var_names_make_unique()
    else:
        adata.var["gene_name"] = adata.var_names.astype(str)

    return adata


def main():
    check_inputs()

    print("[read 10X matrices]")
    a8 = read_one_10x("pbmc8k", PBMC8K_TAR, "0")
    a4 = read_one_10x("pbmc4k", PBMC4K_TAR, "1")

    print("[concat batches]")
    adata = ad.concat(
        [a8, a4],
        join="inner",
        merge="first",
        label="batch",
        keys=["0", "1"],
        index_unique="-batch",
    )
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    print("[raw concatenated]", adata)
    print("[raw batch counts]")
    print(adata.obs["batch"].value_counts())

    print("[attach metadata]")
    adata = attach_metadata(adata)

    print("[filter genes]")
    adata = filter_genes(adata)

    print("[store counts]")
    adata.layers["counts"] = adata.X.copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT_PATH)

    print("\n[done]")
    print("saved:", OUT_PATH)
    print(adata)
    print("\nbatch counts:")
    print(adata.obs["batch"].value_counts())
    print("\ncelltype counts:")
    print(adata.obs["celltype"].value_counts())
    print("\nobs columns:", list(adata.obs.columns))
    print("var columns:", list(adata.var.columns))
    print("layers:", list(adata.layers.keys()))


if __name__ == "__main__":
    main()
