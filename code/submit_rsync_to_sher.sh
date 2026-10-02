#!/bin/bash
#
# Rsync a muchisimocks library dir from Hyperion /data -> Sherlock oak.
#
# Long copies use the Sherlock data-transfer node (dtn.sherlock.stanford.edu).
# Login nodes drop multi-hour sessions; once that multiplexed master dies,
# BatchMode cannot Duo again and every remaining directory fails.
#
# The DTN accepts rsync but not a remote shell, so this script opens the
# ControlMaster with a tiny rsync (password + Duo once) instead of ssh -fN:
#   bash code/submit_rsync_to_sher.sh --parallel 4
#
# Re-run is safe (rsync resumes). Prefer --parallel 4: Sherlock mux caps
# concurrent sessions; higher values can abort mid-transfer.
#
# Watch:
#   tail -f code/logs/rsync_*.log
#
# Close the SSH master later (optional):
#   ssh -O exit sher-dtn

set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${CODE_DIR}/logs"

#SRC_NAME="muchisimocks_lib_coverage_p5_n1000"
#SRC_NAME="muchisimocks_lib_shame_p0_n1000"
SRC_NAME="muchisimocks_lib_p5_n10000"
DATA_ROOT="/data/kstoreyf/muchisimocks"
DEST_HOST="sher-dtn"
DEST_PATH="/oak/stanford/orgs/kipac/data/muchisimocks/"
PARALLEL=4
DRY_RUN=false
RUN_MODE=false

usage() {
  cat <<'EOF'
Usage: submit_rsync_to_sher.sh [options]

  bash code/submit_rsync_to_sher.sh --parallel 4

The DTN has no shell. If no SSH master is open, the script prompts for
password/Duo once via a tiny rsync, then transfers in the background.

Options:
  --src NAME       Dir under /data/kstoreyf/muchisimocks
  --dest-host H    SSH host alias (default: sher-dtn)
  --dest-path P    Remote parent dir (default: /oak/stanford/orgs/kipac/data/muchisimocks/)
  --parallel N     Parallel streams over subdirs (default 4)
  --dry-run        rsync --dry-run
  --run            Internal: do the transfer (used by nohup)
  --help           Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC_NAME="${2:?}"; shift 2 ;;
    --dest-host) DEST_HOST="${2:?}"; shift 2 ;;
    --dest-path) DEST_PATH="${2:?}"; shift 2 ;;
    --parallel) PARALLEL="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --run) RUN_MODE=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

SRC="${DATA_ROOT}/${SRC_NAME}"
REMOTE_DIR="${DEST_PATH%/}/${SRC_NAME}"

dest_uses_dtn() {
  local resolved
  resolved="$(ssh -G "${DEST_HOST}" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }')"
  [[ "${resolved}" == *dtn.sherlock.stanford.edu* || "${DEST_HOST}" == *dtn* ]]
}

# DTN: master must already exist (nohup cannot Duo). Login: remote shell is fine.
require_master_or_shell() {
  local allow_prompt="$1"
  if dest_uses_dtn; then
    if ssh -O check "${DEST_HOST}" >/dev/null 2>&1; then
      echo "Preflight: SSH master ok (${DEST_HOST})."
      return 0
    fi
    if [[ "${allow_prompt}" != true ]]; then
      echo "ERROR: SSH master for ${DEST_HOST} is gone (BatchMode cannot Duo)." >&2
      echo "Re-open it in a terminal, then re-run this script to resume:" >&2
      echo "  bash ${CODE_DIR}/submit_rsync_to_sher.sh --src ${SRC_NAME} --dest-host ${DEST_HOST} --parallel ${PARALLEL}" >&2
      return 1
    fi
    local controlmaster
    controlmaster="$(ssh -G "${DEST_HOST}" 2>/dev/null | awk '$1 == "controlmaster" { print $2; exit }')"
    if [[ "${controlmaster}" != "yes" && "${controlmaster}" != "auto" && "${controlmaster}" != "autoask" ]]; then
      echo "ERROR: ${DEST_HOST} has no ControlMaster in ~/.ssh/config." >&2
      echo "Add a Host ${DEST_HOST} block with ControlMaster auto and ControlPersist 12h." >&2
      return 1
    fi
    echo "Preflight: opening SSH master via rsync (password/Duo once)..."
    local probe_dir rsync_rc
    probe_dir="$(mktemp -d)"
    rsync_rc=0
    # Empty dir: creates REMOTE_DIR if the Oak parent exists, and leaves a
    # ControlPersist master. DTN rejects ssh -fN / remote shells.
    rsync -a -e ssh "${probe_dir}/" "${DEST_HOST}:${REMOTE_DIR}/" || rsync_rc=$?
    rm -rf "${probe_dir}"
    if [[ "${rsync_rc}" -ne 0 ]]; then
      echo "ERROR: could not open an rsync connection to ${DEST_HOST}." >&2
      echo "If the Oak parent is missing, create it once from a login node:" >&2
      echo "  ssh sher \"mkdir -p '${DEST_PATH}'\"" >&2
      return 1
    fi
    if ! ssh -O check "${DEST_HOST}" >/dev/null 2>&1; then
      echo "ERROR: rsync connected but left no ControlMaster for ${DEST_HOST}." >&2
      echo "Check ControlPersist on Host ${DEST_HOST} in ~/.ssh/config." >&2
      return 1
    fi
    echo "Preflight: SSH master ok (${DEST_HOST})."
    return 0
  fi

  if [[ "${allow_prompt}" == true ]]; then
    echo "Preflight: SSH to ${DEST_HOST}..."
    if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "${DEST_HOST}" "echo ok"; then
      echo "ERROR: no reusable SSH session to ${DEST_HOST}." >&2
      echo "Type password/Duo once, then retry:" >&2
      echo "  ssh -fN ${DEST_HOST}" >&2
      return 1
    fi
    echo "Preflight: SSH ok."
    return 0
  fi

  if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "${DEST_HOST}" "mkdir -p '${REMOTE_DIR}' && echo ok"; then
    echo "ERROR: SSH to ${DEST_HOST} failed (BatchMode)." >&2
    echo "Open a multiplexed master first (type password/Duo once):" >&2
    echo "  ssh -fN ${DEST_HOST}" >&2
    return 1
  fi
}

# --- worker ---
if [[ "${RUN_MODE}" == true ]]; then
  echo "========================================"
  echo "Started:   $(date -Is)"
  echo "Host:      $(hostname)"
  echo "Source:    ${SRC}"
  echo "Dest:      ${DEST_HOST}:${REMOTE_DIR}/"
  echo "Parallel:  ${PARALLEL}"
  echo "Dry-run:   ${DRY_RUN}"
  echo "========================================"

  if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: source missing: ${SRC}" >&2
    exit 1
  fi

  require_master_or_shell false

  # No -z: .npy floats barely compress and burn CPU.
  # --partial --append-verify: resume after network drops.
  # ServerAlive*: used if this ssh creates the master. An existing master
  # keeps the intervals from ~/.ssh/config (slave -o flags do not apply).
  RSYNC_SSH='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=60 -c aes128-gcm@openssh.com'
  # Less chatty progress when parallel (interleaved bars are useless).
  INFO_OPTS=(--info=stats2)
  if [[ "${PARALLEL}" -le 1 ]]; then
    INFO_OPTS=(--info=stats2,progress2)
  fi
  RSYNC_OPTS=(-a -h --partial --append-verify "${INFO_OPTS[@]}" -e "${RSYNC_SSH}")
  if [[ "${DRY_RUN}" == true ]]; then
    RSYNC_OPTS+=(--dry-run)
  fi

  rc=0
  if [[ "${PARALLEL}" -le 1 ]]; then
    rsync "${RSYNC_OPTS[@]}" "${SRC}/" "${DEST_HOST}:${REMOTE_DIR}/" || rc=$?
  else
    mapfile -t SUBDIRS < <(find "${SRC}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    echo "Parallel rsync of ${#SUBDIRS[@]} subdirs with ${PARALLEL} streams..."
    FAIL_FILE="$(mktemp)"
    OK_FILE="$(mktemp)"
    # Job pool: continue after failures (unlike xargs, which aborts).
    running=0
    for d in "${SUBDIRS[@]}"; do
      (
        if rsync "${RSYNC_OPTS[@]}" "${SRC}/${d}/" "${DEST_HOST}:${REMOTE_DIR}/${d}/"; then
          echo "${d}" >>"${OK_FILE}"
        else
          echo "${d}" >>"${FAIL_FILE}"
          echo "FAILED: ${d}" >&2
        fi
      ) &
      running=$((running + 1))
      if [[ "${running}" -ge "${PARALLEL}" ]]; then
        wait -n || true
        running=$((running - 1))
      fi
    done
    wait || true

    n_ok="$(wc -l <"${OK_FILE}")"
    n_fail="$(wc -l <"${FAIL_FILE}")"
    echo "Subdir results: ok=${n_ok}  failed=${n_fail}  total=${#SUBDIRS[@]}"
    if [[ "${n_fail}" -gt 0 ]]; then
      rc=1
      echo "Failed subdirs:" >&2
      sort "${FAIL_FILE}" >&2
      echo "Re-run the same command to resume; already-copied files are skipped." >&2
    fi
    rm -f "${OK_FILE}" "${FAIL_FILE}"

    if find "${SRC}" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .; then
      rsync "${RSYNC_OPTS[@]}" --exclude='*/' "${SRC}/" "${DEST_HOST}:${REMOTE_DIR}/" || rc=$?
    fi
  fi

  echo "========================================"
  if [[ "${rc}" -eq 0 ]]; then
    echo "SUCCESS:  $(date -Is)"
  else
    echo "FAILED:   $(date -Is) (exit ${rc})  << incomplete — re-run to resume"
  fi
  echo "========================================"
  exit "${rc}"
fi

# --- launch via nohup ---
if [[ ! -d "${SRC}" ]]; then
  echo "ERROR: source does not exist: ${SRC}" >&2
  exit 1
fi

require_master_or_shell true

ARGS=(--run --src "${SRC_NAME}" --dest-host "${DEST_HOST}" --dest-path "${DEST_PATH}" --parallel "${PARALLEL}")
if [[ "${DRY_RUN}" == true ]]; then
  ARGS+=(--dry-run)
fi

JOB_NAME="rsync_${SRC_NAME}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${CODE_DIR}/logs/${JOB_NAME}_${STAMP}.log"
nohup bash "${CODE_DIR}/submit_rsync_to_sher.sh" "${ARGS[@]}" >"${LOG_FILE}" 2>&1 &
echo "Started transfer (PID $!)"
echo "Log: ${LOG_FILE}"
echo "Watch: tail -f ${LOG_FILE}"
