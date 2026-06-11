#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
META_DIR="$PROJECT_ROOT/data/processed/manifests"
COMMON_DIR="$PROJECT_ROOT/data/processed/common_250hz"

mkdir -p "$META_DIR" "$COMMON_DIR"/{zuco,thinking_out_loud,chisco}

cat > "$META_DIR/subjects.tsv" <<'TSV'
dataset	subject_id	session_id	original_subject_id	notes
TSV

cat > "$META_DIR/sessions.tsv" <<'TSV'
dataset	subject_id	session_id	task_type	sampling_rate_original	sampling_rate_target	recording_system	notes
TSV

cat > "$META_DIR/channels.tsv" <<'TSV'
dataset	subject_id	session_id	channel_name	channel_type	x	y	z	region	keep
TSV

cat > "$META_DIR/trials.csv" <<'CSV'
dataset,subject_id,session_id,trial_id,condition,label_text,label_type,start_sample,end_sample,start_sec,end_sec,split,source_file
CSV

cat > "$META_DIR/label_map.json" <<'JSON'
{

  "zuco": {
    "task": "natural_reading_language_pretraining",
    "labels": ["sentence_text", "word_fixation", "reading_task"],
    "heads": ["semantic_alignment", "reading_task_classification", "sentence_embedding_retrieval"]
  },
  "thinking_out_loud": {
    "task": "inner_speech_command_classification",
    "labels": ["command", "inner_speech", "pronounced", "visualized"],
    "heads": ["command_classification", "condition_classification"]
  },
  "chisco": {
    "task": "sentence_level_imagined_speech",
    "labels": ["sentence_text", "semantic_category", "semantic_anchor"],
    "heads": ["semantic_category", "retrieval", "generation"]
  }
}
JSON

cat > "$META_DIR/common_preprocessing_policy.json" <<'JSON'
{
  "target_sampling_rate_hz": 250,
  "filtering": {
    "notch_hz": [50],
    "bandpass_hz": [0.5, 45.0],
    "alternate_bandpass_hz": [0.5, 70.0]
  },
  "referencing": "common_average_reference",
  "normalization": "zscore_per_subject_session_channel",
  "artifact_policy": {
    "light": ["bad_channel_detection", "interpolation", "eog_regression_if_available"],
    "heavy": ["bad_channel_detection", "ica_or_asr", "manual_qc_flags"]
  },
  "channel_harmonization": {
    "baseline": "common_channel_intersection",
    "recommended": "region_pooling_plus_graph_encoder",
    "advanced": "variable_channel_graph_encoder"
  },
  "feature_representations": {
    "raw": "C x T",
    "wavelet": "C x F x Tprime",
    "graph": "nodes=C, edges=distance_or_connectivity",
    "region": "R x F x Tprime"
  }
}
JSON

echo "[INFO] Common-ground metadata templates created in $META_DIR"
echo "[INFO] Next implement Python loaders to fill these files from detected manifests."
