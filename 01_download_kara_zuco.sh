#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${1:-data/raw}"
EXTERNAL_ROOT="${2:-data/external}"
DOWNLOAD_ZUCO_FULL="${DOWNLOAD_ZUCO_FULL:-0}"

KARA_DIR="$RAW_ROOT/karaone"
ZUCO_DIR="$RAW_ROOT/zuco"
ZUCO_EXTERNAL="$EXTERNAL_ROOT/zuco_benchmark"

mkdir -p "$KARA_DIR" "$ZUCO_DIR" "$ZUCO_EXTERNAL"

echo "[INFO] RAW_ROOT=$RAW_ROOT"
echo "[INFO] EXTERNAL_ROOT=$EXTERNAL_ROOT"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Missing command: $1"
    exit 1
  }
}

need_cmd wget
need_cmd git

echo "[INFO] Downloading KARA ONE archives..."

KARA_BASE="https://www.cs.toronto.edu/~complingweb/data/karaOne"

KARA_FILES=(
  "MM05.tar.bz2"
  "MM08.tar.bz2"
  "MM09.tar.bz2"
  "MM10.tar.bz2"
  "MM11.tar.bz2"
  "MM12.tar.bz2"
  "MM14.tar.bz2"
  "MM15.tar.bz2"
  "MM16.tar.bz2"
  "MM18.tar.bz2"
  "MM19.tar.bz2"
  "MM20.tar.bz2"
  "MM21.tar.bz2"
  "P02.tar.bz2"
)

for f in "${KARA_FILES[@]}"; do
  url="$KARA_BASE/$f"
  out="$KARA_DIR/$f"
  if [[ -f "$out" ]]; then
    echo "[SKIP] $out exists"
  else
    echo "[GET] $url"
    wget -c --tries=5 --timeout=60 -O "$out" "$url"
  fi
done

echo "[INFO] Downloading KARA ONE helper scripts/source..."

mkdir -p "$EXTERNAL_ROOT/karaone_src"
KARA_SRC_FILES=(
  "src/split_data.m"
  "src/get_all_features.m"
  "src.zip"
)

for f in "${KARA_SRC_FILES[@]}"; do
  url="$KARA_BASE/$f"
  out="$EXTERNAL_ROOT/karaone_src/$(basename "$f")"
  if [[ -f "$out" ]]; then
    echo "[SKIP] $out exists"
  else
    echo "[GET] $url"
    wget -c --tries=5 --timeout=60 -O "$out" "$url" || echo "[WARN] Could not fetch $url"
  fi
done

echo "[INFO] Preparing ZuCo benchmark repository..."

if [[ ! -d "$ZUCO_EXTERNAL/.git" ]]; then
  git clone https://github.com/norahollenstein/zuco-benchmark.git "$ZUCO_EXTERNAL"
else
  echo "[SKIP] ZuCo benchmark repo already exists"
  git -C "$ZUCO_EXTERNAL" pull --ff-only || true
fi

cat > "$ZUCO_DIR/README_DOWNLOAD.txt" <<'TXT'
ZuCo data access notes:

ZuCo 1.0 OSF project:
  https://osf.io/q3zws/

ZuCo 2.0 OSF project:
  https://osf.io/2urht/

The official ZuCo benchmark repository is cloned under:
  data/external/zuco_benchmark

The repository provides get_data.sh for the full benchmark dataset.
The full dataset is large, so this script runs it only if:
  DOWNLOAD_ZUCO_FULL=1
TXT

if [[ "$DOWNLOAD_ZUCO_FULL" == "1" ]]; then
  echo "[INFO] DOWNLOAD_ZUCO_FULL=1, running official get_data.sh"
  (
    cd "$ZUCO_EXTERNAL"
    bash get_data.sh
  )
  echo "[INFO] ZuCo full download attempted using official script."
else
  echo "[INFO] Skipping full ZuCo download by default."
  echo "[INFO] To download full ZuCo benchmark data:"
  echo "       DOWNLOAD_ZUCO_FULL=1 bash scripts/01_download_kara_zuco.sh data/raw data/external"
fi

echo "[INFO] Download script completed."
echo "[INFO] KARA ONE raw archives: $KARA_DIR"
echo "[INFO] ZuCo benchmark repo: $ZUCO_EXTERNAL"
