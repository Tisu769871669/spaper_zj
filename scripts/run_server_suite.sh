#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-spaper}"
SUITE="${SUITE:-core}"
SEEDS="${SEEDS:-42 3407 8888 123}"
RUN_CICIOT="${RUN_CICIOT:-1}"
RUN_LEGACY_ANALYSIS="${RUN_LEGACY_ANALYSIS:-1}"
RUN_FT_CIC="${RUN_FT_CIC:-1}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/outputs/logs/server_runs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/run_server_suite_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"
}

run_cmd() {
  log "RUN: $*"
  "$@" 2>&1 | tee -a "$RUN_LOG"
}

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    exit 1
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

ensure_dirs() {
  mkdir -p outputs/results outputs/models outputs/figures outputs/logs outputs/checkpoints runs
}

record_env() {
  log "Repository: $ROOT_DIR"
  log "Suite: $SUITE"
  log "Seeds: $SEEDS"
  run_cmd python --version
  run_cmd python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    run_cmd nvidia-smi
  fi
}

run_tree_models() {
  local dataset="$1"
  for seed in $SEEDS; do
    run_cmd python src/baselines/hgbt_ids.py --dataset "$dataset" --seed "$seed"
    run_cmd python src/baselines/xgboost_ids.py --dataset "$dataset" --seed "$seed"
    run_cmd python src/baselines/lightgbm_ids.py --dataset "$dataset" --seed "$seed"
  done
}

run_lstm_models() {
  local dataset="$1"
  local epochs="$2"
  for seed in $SEEDS; do
    run_cmd python src/baselines/lstm_ids.py --dataset "$dataset" --seed "$seed" --epochs "$epochs"
  done
}

run_biat_mlp() {
  local dataset="$1"
  local epochs="$2"
  local epsilon="$3"
  local alpha="$4"
  local steps="$5"
  local adv_weight="$6"
  local batch_size="${7:-512}"

  for seed in $SEEDS; do
    run_cmd python src/baselines/bilevel_supervised_ids.py \
      --dataset "$dataset" \
      --seed "$seed" \
      --epochs "$epochs" \
      --batch_size "$batch_size" \
      --epsilon "$epsilon" \
      --alpha "$alpha" \
      --steps "$steps" \
      --adv_weight "$adv_weight"
  done
}

run_biat_ft() {
  local dataset="$1"
  local epochs="$2"
  local epsilon="$3"
  local alpha="$4"
  local steps="$5"
  local adv_weight="$6"
  local batch_size="${7:-512}"

  for seed in $SEEDS; do
    run_cmd python src/baselines/bilevel_fttransformer_ids.py \
      --dataset "$dataset" \
      --seed "$seed" \
      --epochs "$epochs" \
      --batch_size "$batch_size" \
      --epsilon "$epsilon" \
      --alpha "$alpha" \
      --steps "$steps" \
      --adv_weight "$adv_weight"
  done
}

run_core_suite() {
  log "=== Core suite: UNSW-NB15 ==="
  run_tree_models "unsw-nb15"
  run_lstm_models "unsw-nb15" 20
  run_biat_mlp "unsw-nb15" 8 0.02 0.005 2 0.6 512
  run_biat_ft "unsw-nb15" 8 0.02 0.005 2 0.6 512
  run_cmd python scripts/evaluate_main_results.py --dataset unsw-nb15

  log "=== Core suite: CIC-IDS2017-random ==="
  run_tree_models "cic-ids2017-random"
  run_lstm_models "cic-ids2017-random" 20
  run_biat_mlp "cic-ids2017-random" 8 0.02 0.005 2 0.6 512
  if [[ "$RUN_FT_CIC" == "1" ]]; then
    run_biat_ft "cic-ids2017-random" 8 0.02 0.005 2 0.6 512
  fi
  run_cmd python scripts/evaluate_main_results.py --dataset cic-ids2017-random
}

run_supplementary_suite() {
  log "=== Supplementary suite: CICIoT2023-grouped ==="
  run_cmd python src/baselines/hgbt_ids.py --dataset ciciot2023-grouped --seed 42
  run_cmd python src/baselines/xgboost_ids.py --dataset ciciot2023-grouped --seed 42
  run_cmd python src/baselines/lightgbm_ids.py --dataset ciciot2023-grouped --seed 42
  run_cmd python src/baselines/bilevel_supervised_ids.py \
    --dataset ciciot2023-grouped \
    --seed 42 \
    --epochs 5 \
    --batch_size 512 \
    --epsilon 0.01 \
    --alpha 0.002 \
    --steps 2 \
    --adv_weight 0.6
  run_cmd python src/baselines/bilevel_fttransformer_ids.py \
    --dataset ciciot2023-grouped \
    --seed 42 \
    --epochs 5 \
    --batch_size 512 \
    --epsilon 0.01 \
    --alpha 0.002 \
    --steps 2 \
    --adv_weight 0.6
  run_cmd python scripts/evaluate_main_results.py --dataset ciciot2023-grouped
}

run_legacy_analysis() {
  log "=== Legacy support analysis: NSL-KDD ==="
  run_cmd python scripts/evaluate_main_results.py --dataset nsl-kdd
  run_cmd python scripts/evaluate_ablation.py --dataset nsl-kdd
  run_cmd python scripts/evaluate_adversarial_robustness.py --dataset nsl-kdd
}

render_figures() {
  log "=== Rendering figures ==="
  run_cmd python scripts/plot_all_figures.py
}

main() {
  activate_conda
  ensure_dirs
  export PYTHONUNBUFFERED=1
  record_env

  case "$SUITE" in
    core)
      run_core_suite
      ;;
    all)
      run_core_suite
      if [[ "$RUN_CICIOT" == "1" ]]; then
        run_supplementary_suite
      fi
      if [[ "$RUN_LEGACY_ANALYSIS" == "1" ]]; then
        run_legacy_analysis
      fi
      ;;
    supplement)
      run_supplementary_suite
      ;;
    *)
      echo "Unsupported SUITE=$SUITE. Use core / all / supplement." >&2
      exit 1
      ;;
  esac

  render_figures
  log "All requested experiments completed."
}

main "$@"
