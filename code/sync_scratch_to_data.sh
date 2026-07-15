#!/bin/bash
#
# Sync muchisimocks content from scratch -> /data (login-node friendly).
#
# /data is typically NFS and often unavailable from compute nodes; run this
# on a login node. Use --background to detach with nohup.
#
# Examples:
#   # Dry-run: show what would be copied for data/
#   ./sync_scratch_to_data.sh --dry-run
#
#   # Sync data/ in the foreground (ok for small transfers / testing)
#   ./sync_scratch_to_data.sh
#
#   # Detach on a login node (recommended for large transfers)
#   ./sync_scratch_to_data.sh --background
#
#   # Later: also sync noise_fields/
#   ./sync_scratch_to_data.sh --include-noise-fields --background
#
#   # Sync a different relative path under the muchisimocks roots
#   ./sync_scratch_to_data.sh --path results --background
#
#   # Delete files on dest that are gone from source (dangerous; off by default)
#   ./sync_scratch_to_data.sh --delete --background

set -euo pipefail

SCRATCH_ROOT="${MUCHISIMOCKS_SCRATCH:-/scratch/kstoreyf/muchisimocks}"
DATA_ROOT="${MUCHISIMOCKS_DATA_ROOT:-/data/kstoreyf/muchisimocks}"

# Relative path under both roots to sync (default: data)
REL_PATH="data"

DRY_RUN=false
BACKGROUND=false
DELETE=false
# When syncing data/, skip noise_fields/ unless --include-noise-fields is set.
INCLUDE_NOISE_FIELDS=false
EXTRA_RSYNC_ARGS=()

usage() {
  cat <<'EOF'
Usage: sync_scratch_to_data.sh [options]

Sync from SCRATCH_ROOT/REL_PATH/ -> DATA_ROOT/REL_PATH/

When REL_PATH is data (the default), noise_fields/ is excluded so you can
sync everything else first. Pass --include-noise-fields to copy it too.

Options:
  --path REL               Relative path under both roots (default: data)
  --dry-run                Pass --dry-run to rsync (no writes)
  --background             Run via nohup in the background; log to DATA_ROOT/logs/
  --include-noise-fields   Also sync data/noise_fields/ (excluded by default)
  --delete                 Pass --delete to rsync (remove dest files missing on source)
  --help                   Show this help

Environment:
  MUCHISIMOCKS_SCRATCH      Scratch project root
                            (default: /scratch/kstoreyf/muchisimocks)
  MUCHISIMOCKS_DATA_ROOT    Destination project root on /data
                            (default: /data/kstoreyf/muchisimocks)

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      REL_PATH="${2:?--path requires an argument}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --background)
      BACKGROUND=true
      shift
      ;;
    --include-noise-fields)
      INCLUDE_NOISE_FIELDS=true
      shift
      ;;
    --delete)
      DELETE=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_RSYNC_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_RSYNC_ARGS+=("$1")
      shift
      ;;
  esac
done

SRC="${SCRATCH_ROOT}/${REL_PATH}/"
DEST="${DATA_ROOT}/${REL_PATH}/"

# Exclude noise_fields only for the default data/ sync (unless overridden).
EXCLUDE_NOISE=false
if [[ "${REL_PATH}" == "data" && "${INCLUDE_NOISE_FIELDS}" == false ]]; then
  EXCLUDE_NOISE=true
fi

if [[ ! -d "${SCRATCH_ROOT}/${REL_PATH}" ]]; then
  echo "ERROR: source does not exist: ${SCRATCH_ROOT}/${REL_PATH}" >&2
  exit 1
fi

# Sanity: warn if we look like a compute node (heuristic).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "WARNING: SLURM_JOB_ID=${SLURM_JOB_ID} — you appear to be on a compute node." >&2
  echo "         /data is often unavailable there; prefer a login node." >&2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: destination root not accessible: ${DATA_ROOT}" >&2
  echo "       If /data is missing, you are probably on a compute node — use a login node." >&2
  exit 1
fi

mkdir -p "${DEST}"

RSYNC_OPTS=(-a -h --partial --info=stats2,progress2)
if [[ "${DRY_RUN}" == true ]]; then
  RSYNC_OPTS+=(--dry-run)
fi
if [[ "${DELETE}" == true ]]; then
  RSYNC_OPTS+=(--delete)
fi
if [[ "${EXCLUDE_NOISE}" == true ]]; then
  # Directory itself and everything under it.
  RSYNC_OPTS+=(--exclude=noise_fields/ --exclude=noise_fields)
fi

run_rsync() {
  echo "========================================"
  echo "Started:  $(date -Is)"
  echo "Host:     $(hostname)"
  echo "Source:   ${SRC}"
  echo "Dest:     ${DEST}"
  echo "Dry-run:  ${DRY_RUN}"
  echo "Delete:   ${DELETE}"
  echo "Exclude noise_fields: ${EXCLUDE_NOISE}"
  echo "========================================"
  # Trailing slashes: sync contents of SRC into DEST.
  rc=0
  rsync "${RSYNC_OPTS[@]}" "${EXTRA_RSYNC_ARGS[@]}" "${SRC}" "${DEST}" || rc=$?
  echo "========================================"
  echo "Finished: $(date -Is) (exit ${rc})"
  echo "========================================"
  return "${rc}"
}

if [[ "${BACKGROUND}" == true ]]; then
  if [[ "${DRY_RUN}" == true ]]; then
    echo "NOTE: --background with --dry-run still detaches; check the log for the dry-run output."
  fi
  LOG_DIR="${DATA_ROOT}/logs"
  mkdir -p "${LOG_DIR}"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SAFE_PATH="$(echo "${REL_PATH}" | tr '/' '_')"
  LOG_FILE="${LOG_DIR}/sync_scratch_to_data_${SAFE_PATH}_${STAMP}.log"
  # Re-exec ourselves without --background so the child does the real work.
  # Rebuild argv from current flags.
  CHILD_ARGS=(--path "${REL_PATH}")
  if [[ "${DRY_RUN}" == true ]]; then
    CHILD_ARGS+=(--dry-run)
  fi
  if [[ "${DELETE}" == true ]]; then
    CHILD_ARGS+=(--delete)
  fi
  if [[ "${INCLUDE_NOISE_FIELDS}" == true ]]; then
    CHILD_ARGS+=(--include-noise-fields)
  fi
  if [[ ${#EXTRA_RSYNC_ARGS[@]} -gt 0 ]]; then
    CHILD_ARGS+=(-- "${EXTRA_RSYNC_ARGS[@]}")
  fi
  nohup "$0" "${CHILD_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
  echo "Started background sync (PID $!)"
  echo "Log: ${LOG_FILE}"
  echo "Watch with:  tail -f ${LOG_FILE}"
  exit 0
fi

run_rsync
