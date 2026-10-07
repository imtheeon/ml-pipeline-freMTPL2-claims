#!/usr/bin/env bash
# Runs the whole pipeline from the repo root. About 3 minutes on a laptop CPU, plus a 9 MB download.
set -e
mkdir -p ml_pipeline/data
for s in 00_get_data 01_eda 02_prep 03_model 04_error_analysis 05_final_exam 06_pricing_view; do
  echo "== $s"; python ml_pipeline/$s.py
done
