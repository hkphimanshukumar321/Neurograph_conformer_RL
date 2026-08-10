#!/bin/bash
# ==============================================================================
# 05_run_all_end_to_end.sh
#
# True end-to-end pipeline for NeuroGraph-Conformer-RL:
#   Step 0: Download datasets
#   Step 1: Scan & detect files
#   Step 2: Build manifests (trials.csv)
#   Step 3: Prepare metadata templates
#   Step 4: Fix / validate trials.csv
#   Step 5: Preprocess & cache EEG tensors
#   Step 6: Training phases 1 → 2 → 3
#
# Self-daemonizing: runs itself under nohup automatically.
#
# Usage:
#   bash scripts/05_run_all_end_to_end.sh [DATASET] [PROTOCOL] [--fg]
#
# Examples:
#   bash scripts/05_run_all_end_to_end.sh                        # defaults, background
#   bash scripts/05_run_all_end_to_end.sh chisco within_subject  # background
#   bash scripts/05_run_all_end_to_end.sh chisco within_subject --fg  # foreground
#
# Monitor progress:
#   tail -f pipeline.log
# ==============================================================================
set -euo pipefail

# ── Resolve paths ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

LOG_FILE="$PROJECT_ROOT/pipeline.log"

# ── Parse arguments ──
DATASET="${1:-chisco}"
PROTOCOL="${2:-within_subject}"
FOREGROUND=0
for arg in "$@"; do
  case "$arg" in
    --fg|--foreground) FOREGROUND=1 ;;
  esac
done

# ── Self-daemonize under nohup (unless already daemonized or --fg) ──
if [[ "$FOREGROUND" == "0" && "${_PIPELINE_DAEMONIZED:-}" != "1" ]]; then
  export _PIPELINE_DAEMONIZED=1
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  Pipeline launched in background via nohup                  ║"
  echo "║  PID will be printed below                                  ║"
  echo "║  Monitor:  tail -f $LOG_FILE  ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  nohup bash "$0" "$DATASET" "$PROTOCOL" --fg > "$LOG_FILE" 2>&1 &
  BGPID=$!
  echo "[INFO] Background PID: $BGPID"
  echo "$BGPID" > "$PROJECT_ROOT/.pipeline_pid"
  exit 0
fi

# ── From here on we are running (either daemonized or --fg) ──

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "======================================================================"
echo "  NeuroGraph-Conformer-RL — Full End-to-End Pipeline"
echo "  Started: $TIMESTAMP"
echo "  Dataset: $DATASET"
echo "  Protocol: $PROTOCOL"
echo "  Project Root: $PROJECT_ROOT"
echo "======================================================================"
echo ""

step_banner() {
  local step_num="$1"
  local step_name="$2"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Step $step_num: $step_name"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── Step 0: Download Datasets ──
step_banner "0/6" "Download Datasets"
if [[ -f "$PROJECT_ROOT/scripts/00_download_all_datasets.sh" ]]; then
  bash "$PROJECT_ROOT/scripts/00_download_all_datasets.sh" || {
    echo "[WARN] Download script encountered errors (may be expected if datasets exist). Continuing..."
  }
else
  echo "[WARN] 00_download_all_datasets.sh not found — skipping download step."
fi

# ── Step 1: Scan & Detect Files ──
step_banner "1/6" "Scan & Detect Files"
if [[ -f "$PROJECT_ROOT/scripts/01_scan_datasets.sh" ]]; then
  bash "$PROJECT_ROOT/scripts/01_scan_datasets.sh"
else
  echo "[WARN] 01_scan_datasets.sh not found — skipping scan step."
fi

# ── Step 2: Prepare Metadata Templates ──
# NOTE: This must run BEFORE build_manifests.py so the templates
# get created first and then overwritten with real data.
step_banner "2/6" "Prepare Metadata Templates"
if [[ -f "$PROJECT_ROOT/scripts/02_prepare_metadata.sh" ]]; then
  bash "$PROJECT_ROOT/scripts/02_prepare_metadata.sh"
else
  echo "[WARN] 02_prepare_metadata.sh not found — skipping metadata prep."
fi

# ── Step 3: Build Manifests (trials.csv) ──
step_banner "3/6" "Build Manifests"
if [[ -f "$PROJECT_ROOT/scripts/build_manifests.py" ]]; then
  python "$PROJECT_ROOT/scripts/build_manifests.py"
else
  echo "[WARN] build_manifests.py not found — skipping manifest build."
fi

# ── Step 4: Fix / Validate trials.csv ──
step_banner "4/6" "Fix & Validate Trials"
if [[ -f "$PROJECT_ROOT/scripts/fix_trials.py" ]]; then
  python "$PROJECT_ROOT/scripts/fix_trials.py"
else
  echo "[WARN] fix_trials.py not found — skipping trial fixes."
fi

# ── Verify trials.csv exists before continuing ──
TRIALS_CSV="$PROJECT_ROOT/data/processed/manifests/trials.csv"
if [[ ! -f "$TRIALS_CSV" ]]; then
  echo "[ERROR] trials.csv still missing after preprocessing steps!"
  echo "[ERROR] Path: $TRIALS_CSV"
  echo "[ERROR] Cannot proceed to preprocessing & training."
  exit 1
fi
TRIAL_COUNT=$(tail -n +2 "$TRIALS_CSV" | wc -l)
echo "[INFO] trials.csv contains $TRIAL_COUNT trials."

# ── Step 5: Preprocess & Cache EEG Tensors ──
step_banner "5/6" "Preprocess & Cache EEG Tensors"
if [[ -f "$PROJECT_ROOT/scripts/03_preprocess_and_cache.py" ]]; then
  python "$PROJECT_ROOT/scripts/03_preprocess_and_cache.py" --project_root "$PROJECT_ROOT"
else
  echo "[WARN] 03_preprocess_and_cache.py not found — skipping caching."
fi

# ── Step 6: Training Phases 1 → 2 → 3 ──
step_banner "6/6" "Training (Phases 1 → 2 → 3)"

echo ""
echo "  ┌─ Phase 1: Supervised Pre-training ─┐"
if [[ -f "$PROJECT_ROOT/scripts/train_phase1.sh" ]]; then
  bash "$PROJECT_ROOT/scripts/train_phase1.sh" "$DATASET" "$PROTOCOL"
else
  echo "[WARN] train_phase1.sh not found — skipping Phase 1."
fi

echo ""
echo "  ┌─ Phase 2: RL Fine-tuning ─┐"
if [[ -f "$PROJECT_ROOT/scripts/train_phase2.sh" ]]; then
  bash "$PROJECT_ROOT/scripts/train_phase2.sh" "$DATASET" "$PROTOCOL"
else
  echo "[WARN] train_phase2.sh not found — skipping Phase 2."
fi

echo ""
echo "  ┌─ Phase 3: Final Evaluation ─┐"
if [[ -f "$PROJECT_ROOT/scripts/train_phase3.sh" ]]; then
  bash "$PROJECT_ROOT/scripts/train_phase3.sh" "$DATASET" "$PROTOCOL"
else
  echo "[WARN] train_phase3.sh not found — skipping Phase 3."
fi

# ── Summary ──
END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "======================================================================"
echo "  ✅ Full Pipeline Completed Successfully!"
echo "  Started:  $TIMESTAMP"
echo "  Finished: $END_TIMESTAMP"
echo "  Dataset:  $DATASET"
echo "  Protocol: $PROTOCOL"
echo "  Results:  experiments/phase3_${DATASET}/"
echo "======================================================================"
