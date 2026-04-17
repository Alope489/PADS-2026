#!/usr/bin/env bash
# rate_gen -> exec -> flow_parser; move artifacts to multiflow/results and plots to results/plots.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_RATES="${SCRIPT_DIR}/scripts/flow.rates"
PLOTS_DIR="${SCRIPT_DIR}/results/plots"
STORAGE_BASE="${SCRIPT_DIR}/multiflow/results"

run_one() {
  local rates_file="$1"
  local run_id="$2"
  if [[ ! -f "$rates_file" ]]; then
    echo "run_all.sh: not a file: $rates_file" >&2
    return 1
  fi
  shopt -s globstar
  export ARTIFACT_ROOT="$SCRIPT_DIR"
  (cd "${SCRIPT_DIR}/scripts" && python3 rate_gen.py "$(cd "$(dirname "$rates_file")" && pwd)/$(basename "$rates_file")") || return 1
  (cd "${SCRIPT_DIR}/multiflow" && ./exec.sh) || return 1
  if ! compgen -G "${SCRIPT_DIR}/multiflow/**/terminal-packet-stats-*" > /dev/null; then
    echo "run_all.sh: replay produced no terminal-packet-stats-* files under multiflow/. Aborting." >&2
    return 1
  fi
  terminal_file=""
  if [[ -d "${SCRIPT_DIR}/multiflow/riodir" ]]; then
    for f in "${SCRIPT_DIR}/multiflow/riodir"/*; do
      if [[ -f "$f" ]]; then
        terminal_file="$f"
        break
      fi
    done
  fi
  if [[ -z "$terminal_file" ]]; then
    terminal_file="$(
      shopt -s nullglob globstar
      for f in "${SCRIPT_DIR}/multiflow"/**/terminal-packet-stats-*; do
        if [[ "$f" == "${SCRIPT_DIR}/multiflow/results/"* ]]; then
          continue
        fi
        if [[ -f "$f" ]]; then
          printf '%s' "$f"
          break
        fi
      done
    )"
  fi
  plot_tmp="${SCRIPT_DIR}/multiflow/plot_${run_id}.png"
  if [[ -f "$terminal_file" ]]; then
    echo "run_all.sh: generating plot from $terminal_file"
    python3 "${SCRIPT_DIR}/scripts/flow_parser.py" "$terminal_file" -o "$plot_tmp" || echo "run_all.sh: flow_parser failed (see above)" >&2
  else
    echo "run_all.sh: no terminal file (riodir/* or terminal-packet-stats-*), skipping plot" >&2
  fi
  mkdir -p "$PLOTS_DIR" "${STORAGE_BASE}/${run_id}"
  [[ -f "$plot_tmp" ]] && mv "$plot_tmp" "${PLOTS_DIR}/${run_id}.png"
  for item in "${SCRIPT_DIR}/multiflow"/*; do
    if [[ ! -e "$item" ]]; then
      continue
    fi
    base="$(basename "$item")"
    if [[ "$base" == "exec.sh" || "$base" == "period.file" || "$base" == "results" ]]; then
      continue
    fi
    rm -rf "${STORAGE_BASE}/${run_id}/${base}"
    mv "$item" "${STORAGE_BASE}/${run_id}/" || true
  done
  shopt -s nullglob
  for item in "${SCRIPT_DIR}/multiflow"/.*; do
    if [[ ! -e "$item" ]]; then
      continue
    fi
    base="$(basename "$item")"
    if [[ "$base" == "." || "$base" == ".." ]]; then
      continue
    fi
    rm -rf "${STORAGE_BASE}/${run_id}/${base}"
    mv "$item" "${STORAGE_BASE}/${run_id}/" || true
  done
  echo "run_id=$run_id plot=${PLOTS_DIR}/${run_id}.png storage=${STORAGE_BASE}/${run_id}"
}

input="${1:-$DEFAULT_RATES}"
if [[ -f "$input" ]]; then
  run_id="$(basename "$input")"
  run_id="${run_id%.*}"
  run_one "$input" "$run_id"
elif [[ -d "$input" ]]; then
  shopt -s nullglob
  for f in "$input"/*; do
    if [[ ! -f "$f" ]]; then
      continue
    fi
    run_id="$(basename "$f")"
    run_id="${run_id%.*}"
    run_one "$f" "$run_id"
  done
else
  echo "run_all.sh: not a file or directory: $input" >&2
  echo "Usage: $0 [rates_file_or_folder]" >&2
  exit 1
fi
