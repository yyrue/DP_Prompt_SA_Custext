#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"

run_cmd() {
  local title="$1"
  shift
  echo
  echo "============================================================"
  echo "[RUN] $title"
  echo "CMD: $*"
  echo "============================================================"
  "$@"
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo "[FAIL] $title (exit code: $exit_code)"
    return $exit_code
  fi
  echo "[OK] $title"
  return 0
}

run_if_dir_exists() {
  local dir_path="$1"
  local title="$2"
  shift 2
  if [[ -d "$dir_path" ]]; then
    run_cmd "$title" "$@"
    return $?
  fi
  echo "[SKIP] $title (directory not found: $dir_path)"
  return 0
}

main() {
  local failed=0

  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python not found: $PYTHON_BIN"
    echo "You can set PYTHON_BIN, e.g. PYTHON_BIN=python3 ./run_all_statistics.sh"
    exit 1
  fi

  # 1) experiment_results
  run_if_dir_exists "sst2_new/experiment_results" \
    "SST2 main results statistics" \
    "$PYTHON_BIN" "sst2_new/collect_results.py" --result_dir "sst2_new/experiment_results" || failed=1
  run_if_dir_exists "qnli_new/experiment_results" \
    "QNLI main results statistics" \
    "$PYTHON_BIN" "sst2_new/collect_results.py" --result_dir "qnli_new/experiment_results" || failed=1

  # 2) baseline statistics
  run_if_dir_exists "sst2_new/experiment_results" \
    "SST2 baseline statistics" \
    "$PYTHON_BIN" "sst2_new/collect_baseline_statistics.py" --result_dir "sst2_new/experiment_results" || failed=1
  run_if_dir_exists "qnli_new/experiment_results" \
    "QNLI baseline statistics" \
    "$PYTHON_BIN" "sst2_new/collect_baseline_statistics.py" --result_dir "qnli_new/experiment_results" || failed=1

  # 3) experiment_results_sa
  run_if_dir_exists "sst2_new/experiment_results_sa" \
    "SST2 SA statistics" \
    "$PYTHON_BIN" "sst2_new/collect_sa_results.py" --base_result_dir "sst2_new/experiment_results_sa" || failed=1
  run_if_dir_exists "qnli_new/experiment_results_sa" \
    "QNLI SA statistics" \
    "$PYTHON_BIN" "sst2_new/collect_sa_results.py" --base_result_dir "qnli_new/experiment_results_sa" || failed=1

  # 4) attack_results + aggregate by eps
  run_if_dir_exists "sst2_new/attack_results" \
    "SST2 mask attack collect" \
    "$PYTHON_BIN" "sst2_new/collect_attack_results.py" --result_dir "sst2_new/attack_results" --auto_scan || failed=1
  if [[ -f "sst2_new/attack_results/attack_summary.csv" ]]; then
    run_cmd "SST2 mask attack aggregate by eps" \
      "$PYTHON_BIN" "sst2_new/aggregate_attack_summary_by_eps.py" \
      --in "sst2_new/attack_results/attack_summary.csv" \
      --out "sst2_new/attack_results/attack_summary_by_eps.csv" || failed=1
  else
    echo "[SKIP] SST2 mask attack aggregate by eps (missing attack_summary.csv)"
  fi

  run_if_dir_exists "qnli_new/attack_results" \
    "QNLI mask attack collect" \
    "$PYTHON_BIN" "sst2_new/collect_attack_results.py" --result_dir "qnli_new/attack_results" --auto_scan || failed=1
  if [[ -f "qnli_new/attack_results/attack_summary.csv" ]]; then
    run_cmd "QNLI mask attack aggregate by eps" \
      "$PYTHON_BIN" "sst2_new/aggregate_attack_summary_by_eps.py" \
      --in "qnli_new/attack_results/attack_summary.csv" \
      --out "qnli_new/attack_results/attack_summary_by_eps.csv" || failed=1
  else
    echo "[SKIP] QNLI mask attack aggregate by eps (missing attack_summary.csv)"
  fi

  # 5) attack_results_mixed
  run_if_dir_exists "sst2_new/attack_results_mixed" \
    "SST2 mixed attack collect from txt" \
    "$PYTHON_BIN" "sst2_new/collect_mixed_attack_from_txt.py" --mixed-root "sst2_new/attack_results_mixed" || failed=1
  run_if_dir_exists "sst2_new/attack_results_mixed" \
    "SST2 mixed attack aggregate by eps_prime" \
    "$PYTHON_BIN" "sst2_new/aggregate_mixed_attack_by_eps_prime.py" --root "sst2_new/attack_results_mixed" || failed=1

  run_if_dir_exists "qnli_new/attack_results_mixed" \
    "QNLI mixed attack collect from txt" \
    "$PYTHON_BIN" "sst2_new/collect_mixed_attack_from_txt.py" --mixed-root "qnli_new/attack_results_mixed" || failed=1
  run_if_dir_exists "qnli_new/attack_results_mixed" \
    "QNLI mixed attack aggregate by eps_prime" \
    "$PYTHON_BIN" "sst2_new/aggregate_mixed_attack_by_eps_prime.py" --root "qnli_new/attack_results_mixed" || failed=1

  # 6) knn attack (normal + mixed)
  run_if_dir_exists "sst2_new/knn_attack_results" \
    "SST2 KNN attack statistics" \
    "$PYTHON_BIN" "sst2_new/collect_knn_attack_results.py" --result_dir "sst2_new/knn_attack_results" || failed=1
  run_if_dir_exists "sst2_new/knn_attack_results/mixed" \
    "SST2 KNN mixed attack statistics" \
    "$PYTHON_BIN" "sst2_new/collect_knn_attack_results.py" --result_dir "sst2_new/knn_attack_results/mixed" || failed=1

  run_if_dir_exists "qnli_new/knn_attack_results" \
    "QNLI KNN attack statistics" \
    "$PYTHON_BIN" "sst2_new/collect_knn_attack_results.py" --result_dir "qnli_new/knn_attack_results" || failed=1
  run_if_dir_exists "qnli_new/knn_attack_results/mixed" \
    "QNLI KNN mixed attack statistics" \
    "$PYTHON_BIN" "sst2_new/collect_knn_attack_results.py" --result_dir "qnli_new/knn_attack_results/mixed" || failed=1

  echo
  echo "============================================================"
  if [[ $failed -eq 0 ]]; then
    echo "All available statistics jobs finished successfully."
    exit 0
  else
    echo "Finished with errors. Check logs above."
    exit 1
  fi
}

main "$@"
