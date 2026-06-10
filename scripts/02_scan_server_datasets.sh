#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SEARCH_ROOTS="${SEARCH_ROOTS:-$PROJECT_ROOT/data/raw:$PROJECT_ROOT:/mnt:/data:/scratch:/home}"
MAX_DEPTH="${MAX_DEPTH:-6}"

MANIFEST_DIR="$PROJECT_ROOT/data/processed/manifests"
RAW_DIR="$PROJECT_ROOT/data/raw"

mkdir -p "$MANIFEST_DIR" "$RAW_DIR"/{karaone,zuco,thinking_out_loud,chisco}

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] SEARCH_ROOTS=$SEARCH_ROOTS"
echo "[INFO] MAX_DEPTH=$MAX_DEPTH"

CSV="$MANIFEST_DIR/detected_datasets.csv"
JSONL="$MANIFEST_DIR/detected_files.jsonl"

echo "dataset,file_type,path,size_bytes" > "$CSV"
: > "$JSONL"

add_record() {
  local dataset="$1"
  local ftype="$2"
  local path="$3"
  local size
  size=$(stat -c%s "$path" 2>/dev/null || echo 0)
  echo "$dataset,$ftype,$path,$size" >> "$CSV"
  printf '{"dataset":"%s","file_type":"%s","path":"%s","size_bytes":%s}\n' "$dataset" "$ftype" "$path" "$size" >> "$JSONL"
}

scan_root() {
  local root="$1"
  [[ -d "$root" ]] || return 0

  echo "[INFO] Scanning $root"

  # KARA ONE archives and extracted content
  find "$root" -maxdepth "$MAX_DEPTH" -type f \( \
    -name "MM*.tar.bz2" -o \
    -name "P02.tar.bz2" -o \
    -name "*.cnt" -o \
    -name "epoch_inds.mat" -o \
    -name "ID.txt" -o \
    -name "ID_p.txt" \
  \) 2>/dev/null | while read -r p; do
    case "$(basename "$p")" in
      MM*.tar.bz2|P02.tar.bz2) add_record "karaone" "archive" "$p" ;;
      *.cnt) add_record "karaone" "cnt_eeg" "$p" ;;
      epoch_inds.mat) add_record "karaone" "epoch_indices" "$p" ;;
      ID.txt|ID_p.txt) add_record "karaone" "prompt_order" "$p" ;;
    esac
  done

  # ZuCo raw/preprocessed/benchmark files
  find "$root" -maxdepth "$MAX_DEPTH" -type f \( \
    -iname "*zuco*.mat" -o \
    -iname "*.mat" -o \
    -iname "*.h5" -o \
    -iname "*.hdf5" -o \
    -iname "get_data.sh" -o \
    -iname "benchmark_baseline.py" \
  \) 2>/dev/null | while read -r p; do
    lower=$(echo "$p" | tr '[:upper:]' '[:lower:]')
    if [[ "$lower" == *zuco* || "$lower" == *"task1"* || "$lower" == *"task2"* || "$lower" == *"nr"* || "$lower" == *"tsr"* ]]; then
      add_record "zuco" "candidate" "$p"
    fi
  done

  # Chisco / Chinese imagined speech candidates
  find "$root" -maxdepth "$MAX_DEPTH" -type f \( \
    -iname "*chisco*" -o \
    -iname "*chineseeeg*" -o \
    -iname "*imagined*speech*" -o \
    -iname "*semantic*.csv" \
  \) 2>/dev/null | while read -r p; do
    add_record "chisco" "candidate" "$p"
  done

  # Thinking Out Loud / inner speech candidates
  find "$root" -maxdepth "$MAX_DEPTH" -type f \( \
    -iname "*thinking*out*loud*" -o \
    -iname "*inner*speech*" -o \
    -iname "*pronounced*" -o \
    -iname "*visualized*" \
  \) 2>/dev/null | while read -r p; do
    add_record "thinking_out_loud" "candidate" "$p"
  done
}

IFS=':' read -r -a roots <<< "$SEARCH_ROOTS"
for r in "${roots[@]}"; do
  scan_root "$r"
done

echo "[INFO] Detection complete."
echo "[INFO] CSV manifest: $CSV"
echo "[INFO] JSONL manifest: $JSONL"

echo "[INFO] Summary:"
cut -d',' -f1 "$CSV" | tail -n +2 | sort | uniq -c || true

echo "[INFO] Creating symlink index directories..."

while IFS=',' read -r dataset ftype path size; do
  [[ "$dataset" == "dataset" ]] && continue
  [[ -f "$path" ]] || continue

  target_dir="$RAW_DIR/$dataset/_linked"
  mkdir -p "$target_dir"

  base="$(basename "$path")"
  safe_base="${ftype}_${base}"
  ln -sfn "$path" "$target_dir/$safe_base"
done < "$CSV"

echo "[INFO] Symlink index created under:"
echo "  $RAW_DIR/karaone/_linked"
echo "  $RAW_DIR/zuco/_linked"
echo "  $RAW_DIR/thinking_out_loud/_linked"
echo "  $RAW_DIR/chisco/_linked"
