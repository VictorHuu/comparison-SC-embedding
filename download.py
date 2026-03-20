from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ctheodoris/Geneformer",
    local_dir="/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/gene_embeddings/intersect/GF-12L95M",
    local_dir_use_symlinks=False
)