#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash hpc/01_create_project_structure.sh [PROJECT_ROOT]" >&2
}

if (( $# > 1 )); then
  usage
  exit 2
fi

if (( $# == 1 )); then
  project_root="$1"
else
  project_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

if [[ ! -d "$project_root" ]]; then
  echo "ERROR: Project root does not exist: $project_root" >&2
  exit 1
fi

project_root="$(cd "$project_root" && pwd -P)"

case "$project_root" in
  /|/home|/head|/head/NFS|/scratch|/tmp)
    echo "ERROR: Refusing to create a project structure in broad path: $project_root" >&2
    exit 1
    ;;
esac

if [[ ! -d "$project_root/.git" && \
      ! -f "$project_root/mapping-waste-urban-africa.Rproj" ]]; then
  echo "ERROR: Target is not a recognized mapping-waste-urban-africa project: $project_root" >&2
  exit 1
fi

directories=(
  "analysis/00_data_preparation/01_annotations"
  "analysis/00_data_preparation/02_tiling"
  "analysis/00_data_preparation/03_spatial_partition"
  "analysis/00_data_preparation/04_dataset_audit"
  "analysis/01_geobia/01_manual_segments"
  "analysis/01_geobia/02_otb_segments"
  "analysis/01_geobia/03_feature_extraction"
  "analysis/01_geobia/04_model_development"
  "analysis/02_unet/01_training"
  "analysis/02_unet/02_ablation"
  "analysis/02_unet/03_prediction"
  "analysis/03_evaluation/01_overall"
  "analysis/03_evaluation/02_landcover_context"
  "analysis/03_evaluation/03_waste_pile_size"
  "analysis/04_sensitivity/01_spatial_resolution"
  "analysis/04_sensitivity/02_illumination"
  "analysis/05_reporting/01_tables"
  "analysis/05_reporting/02_figures"
  "config"
  "data/raw"
  "data/interim"
  "data/processed"
  "data/splits"
  "docs/provenance"
  "environments"
  "hpc/slurm"
  "logs"
  "outputs/metrics"
  "outputs/models"
  "outputs/predictions"
  "outputs/tables"
  "outputs/figures"
  "src/R"
  "src/python"
  "tests"
)

created=0
existing=0

echo "Project root: $project_root"
echo "Creating missing directories only..."

for relative_path in "${directories[@]}"; do
  target="$project_root/$relative_path"

  if [[ -d "$target" ]]; then
    printf 'EXISTS   %s\n' "$relative_path"
    ((existing += 1))
  else
    mkdir -p -- "$target"
    printf 'CREATED  %s\n' "$relative_path"
    ((created += 1))
  fi
done

echo
echo "Structure ready: $created created, $existing already present."
echo "No files were moved, overwritten, or deleted."

