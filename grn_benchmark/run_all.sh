#!/bin/bash
# scGREAT GRN benchmark - generated 2026-03-18 20:49:07.048721
# Experiments: 4 embeddings x 2 datasets

SCGREAT_DIR="/bigdata2/hyt/projects/scGREAT"
OUTPUT_DIR="/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark"

pip install torch scikit-learn pandas numpy 2>/dev/null

RESULTS_FILE="$OUTPUT_DIR/results_summary.txt"
echo "scGREAT GRN Benchmark Results" > "$RESULTS_FILE"
echo "==============================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  difference_v3 x hESC500 (dim=256)"
echo "=========================================="
cd "$SCGREAT_DIR" && python demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/difference_v3/hESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/difference_v3/hESC500/train.log
echo "difference_v3 x hESC500: done" >> "$RESULTS_FILE"
tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/difference_v3/hESC500/train.log >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  difference_v3 x mESC500 (dim=256)"
echo "=========================================="
cd "$SCGREAT_DIR" && python demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/difference_v3/mESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/difference_v3/mESC500/train.log
echo "difference_v3 x mESC500: done" >> "$RESULTS_FILE"
tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/difference_v3/mESC500/train.log >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  baseline x hESC500 (dim=256)"
echo "=========================================="
cd "$SCGREAT_DIR" && python demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/baseline/hESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/baseline/hESC500/train.log
echo "baseline x hESC500: done" >> "$RESULTS_FILE"
tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/baseline/hESC500/train.log >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  baseline x mESC500 (dim=256)"
echo "=========================================="
cd "$SCGREAT_DIR" && python demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/baseline/mESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/baseline/mESC500/train.log
echo "baseline x mESC500: done" >> "$RESULTS_FILE"
tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/baseline/mESC500/train.log >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  scGPT_human x hESC500 (dim=512)"
echo "=========================================="
cd "$SCGREAT_DIR" && python demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/scGPT_human/hESC500 --embed_size 512 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/scGPT_human/hESC500/train.log
echo "scGPT_human x hESC500: done" >> "$RESULTS_FILE"
tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/scGPT_human/hESC500/train.log >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  scGPT_human x mESC500 (dim=512)"
echo "=========================================="
cd "$SCGREAT_DIR" && python demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/scGPT_human/mESC500 --embed_size 512 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/scGPT_human/mESC500/train.log
echo "scGPT_human x mESC500: done" >> "$RESULTS_FILE"
tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/scGPT_human/mESC500/train.log >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  BioBERT_original x hESC500 (dim=768)"
echo "=========================================="
cd "$SCGREAT_DIR" && python demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/BioBERT_original/hESC500 --embed_size 768 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/BioBERT_original/hESC500/train.log
echo "BioBERT_original x hESC500: done" >> "$RESULTS_FILE"
tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/BioBERT_original/hESC500/train.log >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  BioBERT_original x mESC500 (dim=768)"
echo "=========================================="
cd "$SCGREAT_DIR" && python demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/BioBERT_original/mESC500 --embed_size 768 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/BioBERT_original/mESC500/train.log
echo "BioBERT_original x mESC500: done" >> "$RESULTS_FILE"
tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/BioBERT_original/mESC500/train.log >> "$RESULTS_FILE"

echo ""
echo "All experiments complete!"
echo "Results summary: $RESULTS_FILE"