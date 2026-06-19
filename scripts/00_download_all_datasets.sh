#!/usr/bin/env bash
# ==============================================================================
# 00_download_all_datasets.sh
#
# Master download script for the 3 datasets used in NeuroGraph-Conformer-RL:
#   1. Thinking Out Loud (OpenNeuro ds003626) → data/raw/thinking_out_loud
#   2. Chisco            (OpenNeuro ds005170) → data/raw/chisco
#   3. ZuCo              (OSF / zuco-benchmark repo) → data/raw/zuco
#
# The existing OpenNeuro download scripts (ds003626-2.1.2.sh, ds005170-1.1.2.sh)
# download into versioned folders at CWD. This script wraps them and creates
# symlinks into data/raw/ so the pipeline sees a consistent layout.
#
# Usage:
#   bash scripts/00_download_all_datasets.sh [--dry-run] [--skip-tol] [--skip-chisco] [--skip-zuco]
#
# Environment variables:
#   PROJECT_ROOT        — defaults to the parent of this script's directory
#   DOWNLOAD_ZUCO_FULL  — set to 1 to download full ZuCo .mat files (large)
# ==============================================================================
set -euo pipefail

# ── Resolve PROJECT_ROOT ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

DRY_RUN=0
SKIP_TOL=0
SKIP_CHISCO=0
SKIP_ZUCO=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)    DRY_RUN=1 ;;
    --skip-tol)   SKIP_TOL=1 ;;
    --skip-chisco) SKIP_CHISCO=1 ;;
    --skip-zuco)  SKIP_ZUCO=1 ;;
    *) echo "[WARN] Unknown argument: $arg" ;;
  esac
done

RAW_DIR="$PROJECT_ROOT/data/raw"
mkdir -p "$RAW_DIR"

echo "============================================================"
echo "  NeuroGraph-Conformer-RL — Dataset Download"
echo "============================================================"
echo "[INFO] PROJECT_ROOT = $PROJECT_ROOT"
echo "[INFO] RAW_DIR      = $RAW_DIR"
echo "[INFO] DRY_RUN      = $DRY_RUN"
echo ""

# ──────────────────────────────────────────────────────────────────
# 1. Thinking Out Loud (ds003626) — Inner Speech BDF files
# ──────────────────────────────────────────────────────────────────
if [[ "$SKIP_TOL" == "0" ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [1/3] Thinking Out Loud (ds003626)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  TOL_DOWNLOAD_SCRIPT="$PROJECT_ROOT/ds003626-2.1.2.sh"
  TOL_DOWNLOAD_DIR="$PROJECT_ROOT/ds003626-2.1.2"
  TOL_RAW_DIR="$RAW_DIR/thinking_out_loud"

  if [[ ! -f "$TOL_DOWNLOAD_SCRIPT" ]]; then
    echo "[ERROR] Missing download script: $TOL_DOWNLOAD_SCRIPT"
    echo "[ERROR] Get it from: https://openneuro.org/datasets/ds003626"
    exit 1
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Would execute: bash $TOL_DOWNLOAD_SCRIPT"
    echo "[DRY-RUN] Would symlink: $TOL_DOWNLOAD_DIR → $TOL_RAW_DIR"
  else
    # Check if already downloaded
    BDF_COUNT=0
    if [[ -d "$TOL_DOWNLOAD_DIR" ]]; then
      BDF_COUNT=$(find "$TOL_DOWNLOAD_DIR" -name "*.bdf" 2>/dev/null | wc -l)
    fi

    if [[ "$BDF_COUNT" -ge 10 ]]; then
      echo "[SKIP] Thinking Out Loud already downloaded ($BDF_COUNT .bdf files found)"
    else
      echo "[INFO] Downloading Thinking Out Loud..."
      (cd "$PROJECT_ROOT" && bash "$TOL_DOWNLOAD_SCRIPT")
      echo "[INFO] Download complete."
    fi

    # Create symlink into data/raw/
    if [[ -d "$TOL_DOWNLOAD_DIR" ]]; then
      mkdir -p "$TOL_RAW_DIR"
      # Symlink the BIDS sub-* directories into the raw dir
      for subdir in "$TOL_DOWNLOAD_DIR"/sub-*; do
        if [[ -d "$subdir" ]]; then
          base=$(basename "$subdir")
          if [[ ! -e "$TOL_RAW_DIR/$base" ]]; then
            ln -sfn "$subdir" "$TOL_RAW_DIR/$base"
          fi
        fi
      done
      # Also link the metadata files
      for meta_file in "$TOL_DOWNLOAD_DIR"/*.json "$TOL_DOWNLOAD_DIR"/*.tsv "$TOL_DOWNLOAD_DIR"/README "$TOL_DOWNLOAD_DIR"/CHANGES; do
        if [[ -f "$meta_file" ]]; then
          base=$(basename "$meta_file")
          if [[ ! -e "$TOL_RAW_DIR/$base" ]]; then
            ln -sfn "$meta_file" "$TOL_RAW_DIR/$base"
          fi
        fi
      done
      # Link the derivatives too
      if [[ -d "$TOL_DOWNLOAD_DIR/derivatives" ]] && [[ ! -e "$TOL_RAW_DIR/derivatives" ]]; then
        ln -sfn "$TOL_DOWNLOAD_DIR/derivatives" "$TOL_RAW_DIR/derivatives"
      fi
      echo "[OK] Thinking Out Loud linked to $TOL_RAW_DIR"
    fi
  fi
  echo ""
fi

# ──────────────────────────────────────────────────────────────────
# 2. Chisco (ds005170) — Chinese Imagined Speech EDF files
# ──────────────────────────────────────────────────────────────────
if [[ "$SKIP_CHISCO" == "0" ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [2/3] Chisco (ds005170)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  CHISCO_DOWNLOAD_SCRIPT="$PROJECT_ROOT/ds005170-1.1.2.sh"
  CHISCO_DOWNLOAD_DIR="$PROJECT_ROOT/ds005170-1.1.2"
  CHISCO_RAW_DIR="$RAW_DIR/chisco"

  if [[ ! -f "$CHISCO_DOWNLOAD_SCRIPT" ]]; then
    echo "[ERROR] Missing download script: $CHISCO_DOWNLOAD_SCRIPT"
    echo "[ERROR] Get it from: https://openneuro.org/datasets/ds005170"
    exit 1
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Would execute: bash $CHISCO_DOWNLOAD_SCRIPT"
    echo "[DRY-RUN] Would symlink: $CHISCO_DOWNLOAD_DIR → $CHISCO_RAW_DIR"
  else
    # Check if already downloaded
    EDF_COUNT=0
    if [[ -d "$CHISCO_DOWNLOAD_DIR" ]]; then
      EDF_COUNT=$(find "$CHISCO_DOWNLOAD_DIR" -name "*.edf" 2>/dev/null | wc -l)
    fi

    if [[ "$EDF_COUNT" -ge 10 ]]; then
      echo "[SKIP] Chisco already downloaded ($EDF_COUNT .edf files found)"
    else
      echo "[INFO] Downloading Chisco..."
      (cd "$PROJECT_ROOT" && bash "$CHISCO_DOWNLOAD_SCRIPT")
      echo "[INFO] Download complete."
    fi

    # Create symlink into data/raw/
    if [[ -d "$CHISCO_DOWNLOAD_DIR" ]]; then
      mkdir -p "$CHISCO_RAW_DIR"
      for subdir in "$CHISCO_DOWNLOAD_DIR"/sub-*; do
        if [[ -d "$subdir" ]]; then
          base=$(basename "$subdir")
          if [[ ! -e "$CHISCO_RAW_DIR/$base" ]]; then
            ln -sfn "$subdir" "$CHISCO_RAW_DIR/$base"
          fi
        fi
      done
      # Link metadata files
      for meta_file in "$CHISCO_DOWNLOAD_DIR"/*.json "$CHISCO_DOWNLOAD_DIR"/*.tsv "$CHISCO_DOWNLOAD_DIR"/README "$CHISCO_DOWNLOAD_DIR"/CHANGES; do
        if [[ -f "$meta_file" ]]; then
          base=$(basename "$meta_file")
          if [[ ! -e "$CHISCO_RAW_DIR/$base" ]]; then
            ln -sfn "$meta_file" "$CHISCO_RAW_DIR/$base"
          fi
        fi
      done
      # Link the textdataset folder (Chisco sentence labels)
      if [[ -d "$CHISCO_DOWNLOAD_DIR/textdataset" ]] && [[ ! -e "$CHISCO_RAW_DIR/textdataset" ]]; then
        ln -sfn "$CHISCO_DOWNLOAD_DIR/textdataset" "$CHISCO_RAW_DIR/textdataset"
      fi
      echo "[OK] Chisco linked to $CHISCO_RAW_DIR"
    fi
  fi
  echo ""
fi

# ──────────────────────────────────────────────────────────────────
# 3. ZuCo — Natural Reading EEG (.mat files via OSF benchmark)
# ──────────────────────────────────────────────────────────────────
if [[ "$SKIP_ZUCO" == "0" ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [3/3] ZuCo (zuco-benchmark)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  ZUCO_DOWNLOAD_SCRIPT="$PROJECT_ROOT/01_download_kara_zuco.sh"

  if [[ ! -f "$ZUCO_DOWNLOAD_SCRIPT" ]]; then
    echo "[ERROR] Missing download script: $ZUCO_DOWNLOAD_SCRIPT"
    exit 1
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Would execute: bash $ZUCO_DOWNLOAD_SCRIPT $RAW_DIR $PROJECT_ROOT/data/external"
  else
    echo "[INFO] Downloading ZuCo benchmark + KaraOne..."
    bash "$ZUCO_DOWNLOAD_SCRIPT" "$RAW_DIR" "$PROJECT_ROOT/data/external"
    echo "[OK] ZuCo download complete."
  fi
  echo ""
fi

# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Download Summary"
echo "============================================================"
echo ""

for ds_name in thinking_out_loud chisco zuco; do
  ds_dir="$RAW_DIR/$ds_name"
  if [[ -d "$ds_dir" ]]; then
    file_count=$(find -L "$ds_dir" -type f \( -name "*.bdf" -o -name "*.edf" -o -name "*.mat" -o -name "*.vhdr" \) 2>/dev/null | wc -l)
    echo "  ✓ $ds_name: $file_count EEG files in $ds_dir"
  else
    echo "  ✗ $ds_name: NOT FOUND at $ds_dir"
  fi
done

echo ""
echo "[INFO] Next steps:"
echo "  1. bash scripts/01_scan_datasets.sh"
echo "  2. python scripts/build_manifests.py"
echo "  3. python scripts/fix_trials.py"
echo "  4. python scripts/03_preprocess_and_cache.py --project_root ."
echo ""
echo "  Or run the full pipeline:"
echo "  nohup bash scripts/05_run_all_end_to_end.sh > pipeline.log 2>&1 &"
echo "============================================================"
