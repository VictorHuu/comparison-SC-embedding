#!/bin/bash
# set -uo pipefail
# scGREAT GRN benchmark - generated 2026-04-04 13:37:52.103740
# Experiments: 8 embeddings x 6 datasets

SCGREAT_DIR="/bigdata2/hyt/projects/scGREAT"
OUTPUT_DIR="/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_DIR"

RESULTS_FILE="$OUTPUT_DIR/results_summary.txt"
echo "scGREAT GRN Benchmark Results" > "$RESULTS_FILE"
echo "==============================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  difference_v3 x hESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hESC500/BL--ExpressionData.csv ]; then echo "SKIP difference_v3 x hESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hESC500/train.log; then
  echo "difference_v3 x hESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "difference_v3 x hESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  difference_v3 x hHep500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hHep500/BL--ExpressionData.csv ]; then echo "SKIP difference_v3 x hHep500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hHep500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hHep500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hHep500/train.log; then
  echo "difference_v3 x hHep500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hHep500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/hHep500/train.log >> "$RESULTS_FILE"; fi
else
  echo "difference_v3 x hHep500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  difference_v3 x mESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mESC500/BL--ExpressionData.csv ]; then echo "SKIP difference_v3 x mESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mESC500/train.log; then
  echo "difference_v3 x mESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "difference_v3 x mESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  difference_v3 x mHSC-E500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-E500/BL--ExpressionData.csv ]; then echo "SKIP difference_v3 x mHSC-E500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-E500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-E500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-E500/train.log; then
  echo "difference_v3 x mHSC-E500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-E500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-E500/train.log >> "$RESULTS_FILE"; fi
else
  echo "difference_v3 x mHSC-E500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  difference_v3 x mHSC-GM500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-GM500/BL--ExpressionData.csv ]; then echo "SKIP difference_v3 x mHSC-GM500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-GM500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-GM500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-GM500/train.log; then
  echo "difference_v3 x mHSC-GM500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-GM500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-GM500/train.log >> "$RESULTS_FILE"; fi
else
  echo "difference_v3 x mHSC-GM500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  difference_v3 x mHSC-L500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-L500/BL--ExpressionData.csv ]; then echo "SKIP difference_v3 x mHSC-L500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-L500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-L500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-L500/train.log; then
  echo "difference_v3 x mHSC-L500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-L500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/difference_v3/mHSC-L500/train.log >> "$RESULTS_FILE"; fi
else
  echo "difference_v3 x mHSC-L500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  minus x hESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hESC500/BL--ExpressionData.csv ]; then echo "SKIP minus x hESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hESC500/train.log; then
  echo "minus x hESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "minus x hESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  minus x hHep500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hHep500/BL--ExpressionData.csv ]; then echo "SKIP minus x hHep500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hHep500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hHep500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hHep500/train.log; then
  echo "minus x hHep500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hHep500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/hHep500/train.log >> "$RESULTS_FILE"; fi
else
  echo "minus x hHep500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  minus x mESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mESC500/BL--ExpressionData.csv ]; then echo "SKIP minus x mESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mESC500/train.log; then
  echo "minus x mESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "minus x mESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  minus x mHSC-E500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-E500/BL--ExpressionData.csv ]; then echo "SKIP minus x mHSC-E500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-E500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-E500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-E500/train.log; then
  echo "minus x mHSC-E500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-E500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-E500/train.log >> "$RESULTS_FILE"; fi
else
  echo "minus x mHSC-E500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  minus x mHSC-GM500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-GM500/BL--ExpressionData.csv ]; then echo "SKIP minus x mHSC-GM500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-GM500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-GM500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-GM500/train.log; then
  echo "minus x mHSC-GM500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-GM500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-GM500/train.log >> "$RESULTS_FILE"; fi
else
  echo "minus x mHSC-GM500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  minus x mHSC-L500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-L500/BL--ExpressionData.csv ]; then echo "SKIP minus x mHSC-L500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-L500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-L500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-L500/train.log; then
  echo "minus x mHSC-L500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-L500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/minus/mHSC-L500/train.log >> "$RESULTS_FILE"; fi
else
  echo "minus x mHSC-L500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  baseline x hESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hESC500/BL--ExpressionData.csv ]; then echo "SKIP baseline x hESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hESC500/train.log; then
  echo "baseline x hESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "baseline x hESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  baseline x hHep500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hHep500/BL--ExpressionData.csv ]; then echo "SKIP baseline x hHep500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hHep500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hHep500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hHep500/train.log; then
  echo "baseline x hHep500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hHep500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/hHep500/train.log >> "$RESULTS_FILE"; fi
else
  echo "baseline x hHep500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  baseline x mESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mESC500/BL--ExpressionData.csv ]; then echo "SKIP baseline x mESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mESC500/train.log; then
  echo "baseline x mESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "baseline x mESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  baseline x mHSC-E500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-E500/BL--ExpressionData.csv ]; then echo "SKIP baseline x mHSC-E500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-E500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-E500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-E500/train.log; then
  echo "baseline x mHSC-E500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-E500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-E500/train.log >> "$RESULTS_FILE"; fi
else
  echo "baseline x mHSC-E500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  baseline x mHSC-GM500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-GM500/BL--ExpressionData.csv ]; then echo "SKIP baseline x mHSC-GM500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-GM500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-GM500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-GM500/train.log; then
  echo "baseline x mHSC-GM500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-GM500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-GM500/train.log >> "$RESULTS_FILE"; fi
else
  echo "baseline x mHSC-GM500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  baseline x mHSC-L500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-L500/BL--ExpressionData.csv ]; then echo "SKIP baseline x mHSC-L500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-L500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-L500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-L500/train.log; then
  echo "baseline x mHSC-L500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-L500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/baseline/mHSC-L500/train.log >> "$RESULTS_FILE"; fi
else
  echo "baseline x mHSC-L500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  scGPT_human x hESC500 (dim=512)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hESC500/BL--ExpressionData.csv ]; then echo "SKIP scGPT_human x hESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hESC500 --embed_size 512 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hESC500/train.log; then
  echo "scGPT_human x hESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "scGPT_human x hESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  scGPT_human x hHep500 (dim=512)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hHep500/BL--ExpressionData.csv ]; then echo "SKIP scGPT_human x hHep500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hHep500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hHep500 --embed_size 512 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hHep500/train.log; then
  echo "scGPT_human x hHep500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hHep500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/hHep500/train.log >> "$RESULTS_FILE"; fi
else
  echo "scGPT_human x hHep500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  scGPT_human x mESC500 (dim=512)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mESC500/BL--ExpressionData.csv ]; then echo "SKIP scGPT_human x mESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mESC500 --embed_size 512 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mESC500/train.log; then
  echo "scGPT_human x mESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "scGPT_human x mESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  scGPT_human x mHSC-E500 (dim=512)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-E500/BL--ExpressionData.csv ]; then echo "SKIP scGPT_human x mHSC-E500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-E500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-E500 --embed_size 512 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-E500/train.log; then
  echo "scGPT_human x mHSC-E500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-E500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-E500/train.log >> "$RESULTS_FILE"; fi
else
  echo "scGPT_human x mHSC-E500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  scGPT_human x mHSC-GM500 (dim=512)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-GM500/BL--ExpressionData.csv ]; then echo "SKIP scGPT_human x mHSC-GM500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-GM500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-GM500 --embed_size 512 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-GM500/train.log; then
  echo "scGPT_human x mHSC-GM500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-GM500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-GM500/train.log >> "$RESULTS_FILE"; fi
else
  echo "scGPT_human x mHSC-GM500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  scGPT_human x mHSC-L500 (dim=512)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-L500/BL--ExpressionData.csv ]; then echo "SKIP scGPT_human x mHSC-L500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-L500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-L500 --embed_size 512 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-L500/train.log; then
  echo "scGPT_human x mHSC-L500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-L500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/scGPT_human/mHSC-L500/train.log >> "$RESULTS_FILE"; fi
else
  echo "scGPT_human x mHSC-L500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_bias_rec_best x hESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hESC500/BL--ExpressionData.csv ]; then echo "SKIP v4_bias_rec_best x hESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hESC500/train.log; then
  echo "v4_bias_rec_best x hESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_bias_rec_best x hESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_bias_rec_best x hHep500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hHep500/BL--ExpressionData.csv ]; then echo "SKIP v4_bias_rec_best x hHep500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hHep500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hHep500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hHep500/train.log; then
  echo "v4_bias_rec_best x hHep500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hHep500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/hHep500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_bias_rec_best x hHep500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_bias_rec_best x mESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mESC500/BL--ExpressionData.csv ]; then echo "SKIP v4_bias_rec_best x mESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mESC500/train.log; then
  echo "v4_bias_rec_best x mESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_bias_rec_best x mESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_bias_rec_best x mHSC-E500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-E500/BL--ExpressionData.csv ]; then echo "SKIP v4_bias_rec_best x mHSC-E500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-E500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-E500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-E500/train.log; then
  echo "v4_bias_rec_best x mHSC-E500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-E500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-E500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_bias_rec_best x mHSC-E500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_bias_rec_best x mHSC-GM500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-GM500/BL--ExpressionData.csv ]; then echo "SKIP v4_bias_rec_best x mHSC-GM500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-GM500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-GM500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-GM500/train.log; then
  echo "v4_bias_rec_best x mHSC-GM500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-GM500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-GM500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_bias_rec_best x mHSC-GM500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_bias_rec_best x mHSC-L500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-L500/BL--ExpressionData.csv ]; then echo "SKIP v4_bias_rec_best x mHSC-L500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-L500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-L500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-L500/train.log; then
  echo "v4_bias_rec_best x mHSC-L500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-L500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_bias_rec_best/mHSC-L500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_bias_rec_best x mHSC-L500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_plain_best x hESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hESC500/BL--ExpressionData.csv ]; then echo "SKIP v4_plain_best x hESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hESC500/train.log; then
  echo "v4_plain_best x hESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_plain_best x hESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_plain_best x hHep500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hHep500/BL--ExpressionData.csv ]; then echo "SKIP v4_plain_best x hHep500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hHep500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hHep500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hHep500/train.log; then
  echo "v4_plain_best x hHep500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hHep500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/hHep500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_plain_best x hHep500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_plain_best x mESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mESC500/BL--ExpressionData.csv ]; then echo "SKIP v4_plain_best x mESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mESC500/train.log; then
  echo "v4_plain_best x mESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_plain_best x mESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_plain_best x mHSC-E500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-E500/BL--ExpressionData.csv ]; then echo "SKIP v4_plain_best x mHSC-E500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-E500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-E500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-E500/train.log; then
  echo "v4_plain_best x mHSC-E500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-E500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-E500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_plain_best x mHSC-E500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_plain_best x mHSC-GM500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-GM500/BL--ExpressionData.csv ]; then echo "SKIP v4_plain_best x mHSC-GM500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-GM500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-GM500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-GM500/train.log; then
  echo "v4_plain_best x mHSC-GM500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-GM500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-GM500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_plain_best x mHSC-GM500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_plain_best x mHSC-L500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-L500/BL--ExpressionData.csv ]; then echo "SKIP v4_plain_best x mHSC-L500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-L500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-L500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-L500/train.log; then
  echo "v4_plain_best x mHSC-L500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-L500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_plain_best/mHSC-L500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_plain_best x mHSC-L500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_type_pe_best x hESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hESC500/BL--ExpressionData.csv ]; then echo "SKIP v4_type_pe_best x hESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hESC500/train.log; then
  echo "v4_type_pe_best x hESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_type_pe_best x hESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_type_pe_best x hHep500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hHep500/BL--ExpressionData.csv ]; then echo "SKIP v4_type_pe_best x hHep500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hHep500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hHep500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hHep500/train.log; then
  echo "v4_type_pe_best x hHep500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hHep500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/hHep500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_type_pe_best x hHep500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_type_pe_best x mESC500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mESC500/BL--ExpressionData.csv ]; then echo "SKIP v4_type_pe_best x mESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mESC500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mESC500/train.log; then
  echo "v4_type_pe_best x mESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_type_pe_best x mESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_type_pe_best x mHSC-E500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-E500/BL--ExpressionData.csv ]; then echo "SKIP v4_type_pe_best x mHSC-E500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-E500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-E500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-E500/train.log; then
  echo "v4_type_pe_best x mHSC-E500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-E500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-E500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_type_pe_best x mHSC-E500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_type_pe_best x mHSC-GM500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-GM500/BL--ExpressionData.csv ]; then echo "SKIP v4_type_pe_best x mHSC-GM500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-GM500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-GM500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-GM500/train.log; then
  echo "v4_type_pe_best x mHSC-GM500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-GM500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-GM500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_type_pe_best x mHSC-GM500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  v4_type_pe_best x mHSC-L500 (dim=256)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-L500/BL--ExpressionData.csv ]; then echo "SKIP v4_type_pe_best x mHSC-L500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-L500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-L500 --embed_size 256 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-L500/train.log; then
  echo "v4_type_pe_best x mHSC-L500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-L500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/v4_type_pe_best/mHSC-L500/train.log >> "$RESULTS_FILE"; fi
else
  echo "v4_type_pe_best x mHSC-L500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  BioBERT_original x hESC500 (dim=768)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hESC500/BL--ExpressionData.csv ]; then echo "SKIP BioBERT_original x hESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hESC500 --embed_size 768 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hESC500/train.log; then
  echo "BioBERT_original x hESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "BioBERT_original x hESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  BioBERT_original x hHep500 (dim=768)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hHep500/BL--ExpressionData.csv ]; then echo "SKIP BioBERT_original x hHep500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hHep500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hHep500 --embed_size 768 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hHep500/train.log; then
  echo "BioBERT_original x hHep500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hHep500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/hHep500/train.log >> "$RESULTS_FILE"; fi
else
  echo "BioBERT_original x hHep500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  BioBERT_original x mESC500 (dim=768)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mESC500/BL--ExpressionData.csv ]; then echo "SKIP BioBERT_original x mESC500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mESC500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mESC500 --embed_size 768 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mESC500/train.log; then
  echo "BioBERT_original x mESC500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mESC500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mESC500/train.log >> "$RESULTS_FILE"; fi
else
  echo "BioBERT_original x mESC500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  BioBERT_original x mHSC-E500 (dim=768)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-E500/BL--ExpressionData.csv ]; then echo "SKIP BioBERT_original x mHSC-E500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-E500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-E500 --embed_size 768 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-E500/train.log; then
  echo "BioBERT_original x mHSC-E500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-E500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-E500/train.log >> "$RESULTS_FILE"; fi
else
  echo "BioBERT_original x mHSC-E500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  BioBERT_original x mHSC-GM500 (dim=768)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-GM500/BL--ExpressionData.csv ]; then echo "SKIP BioBERT_original x mHSC-GM500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-GM500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-GM500 --embed_size 768 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-GM500/train.log; then
  echo "BioBERT_original x mHSC-GM500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-GM500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-GM500/train.log >> "$RESULTS_FILE"; fi
else
  echo "BioBERT_original x mHSC-GM500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "=========================================="
echo "  BioBERT_original x mHSC-L500 (dim=768)"
echo "=========================================="
if [ ! -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-L500/BL--ExpressionData.csv ]; then echo "SKIP BioBERT_original x mHSC-L500: missing BL--ExpressionData.csv in /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-L500" | tee -a "$RESULTS_FILE"; continue; fi
if cd "$SCGREAT_DIR" && "$PYTHON_BIN" demo.py --data_dir /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-L500 --embed_size 768 --num_layers 2 --num_head 4 --epochs 50 --batch_size 64 --lr 1e-4 --n_runs 5 2>&1 | tee /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-L500/train.log; then
  echo "BioBERT_original x mHSC-L500: done" >> "$RESULTS_FILE"
  if [ -f /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-L500/train.log ]; then tail -3 /bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark/BioBERT_original/mHSC-L500/train.log >> "$RESULTS_FILE"; fi
else
  echo "BioBERT_original x mHSC-L500: FAILED (continue next)" | tee -a "$RESULTS_FILE"
  continue
fi

echo ""
echo "All experiments complete!"
echo "Results summary: $RESULTS_FILE"