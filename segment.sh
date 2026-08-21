#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 <input-file|input-dir> <output-dir>

Arguments:
  input-file|input-dir  File or directory containing NIfTI files (.nii, .nii.gz)
  output-dir            Directory to write resampled and normalised files

Example:
  $0 dataset/processed/segmentations/
  $0 dataset/processed/scan1.nii.gz segmentations/
USAGE
}

if [ $# -neq 2 ]; then
  usage
  exit 1
fi

input="$1"
output_dir="$2"
model_dir="best_model/"

mkdir -p "$output_dir"

get_output_name() {
  local infile="$1"
  local filename
  filename="$(basename "$infile")"
  local stem

  if [[ "$filename" == *.nii.gz ]]; then
    stem="${filename%.nii.gz}"
  elif [[ "$filename" == *.nii ]]; then
    stem="${filename%.nii}"
  else
    stem="${filename%.*}"
  fi

  printf "%s_seg.nii.gz" "$stem"
}

process_file() {
  local infile="$1"
  local outfile_name
  outfile_name="$(get_output_name "$infile")"
  local outfile_path="$output_dir/$outfile_name"

  python3 segment.py "$model_dir" "$infile" "$outfile_path" 0

  echo "Saved: $outfile_path"
}

if [ -d "$input" ]; then
  shopt -s nullglob
  files=("$input"/*.nii "$input"/*.nii.gz)
  if [ ${#files[@]} -eq 0 ]; then
    echo "No .nii or .nii.gz files found in directory: $input"
    exit 1
  fi
  for i in $(seq 0 $(( ${#files[@]} - 1 ))); do
    infile="${files[$i]}"
    printf "[%s/%s] Processing: %s\n" "$((i + 1))" "${#files[@]}" "$infile"
    process_file "$infile"
  done
else
  process_file "$input"
fi