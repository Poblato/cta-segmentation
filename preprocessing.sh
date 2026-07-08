#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 <input-file|input-dir> <output-dir> [mean] [std]

Arguments:
  input-file|input-dir  File or directory containing NIfTI files (.nii, .nii.gz)
  output-dir            Directory to write resampled and normalised files
  mean                  Target mean for normalisation (default: 0.0)
  std                   Target standard deviation for normalisation (default: 1.0)

Example:
  $0 dataset/raw/ dataset/processed/ 0.0 1.0
  $0 dataset/raw/1-200/scan001.nii.gz dataset/processed/
USAGE
}

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
  usage
  exit 1
fi

input="$1"
output_dir="$2"
target_mean="${3:-0.0}"
target_std="${4:-1.0}"

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

  printf "%s_processed.nii.gz" "$stem"
}

process_file() {
  local infile="$1"
  local outfile_name
  outfile_name="$(get_output_name "$infile")"
  local outfile_path="$output_dir/$outfile_name"
  local tmpfile

  tmpfile="$(mktemp -t resample_tmp.XXXXXX).nii.gz"
  trap 'rm -f "$tmpfile"' RETURN

  python3 preprocessing_scripts/resampling.py 1 128 128 128 "$infile" "$tmpfile"

  python3 preprocessing_scripts/normalisation.py "$target_mean" "$target_std" "$tmpfile" "$outfile_path"

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