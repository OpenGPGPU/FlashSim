#!/usr/bin/env bash
set -euo pipefail
# Profile-guided ARTI build: instrument, train, merge, rebuild.
# Usage:
#   ./scripts/pgo_arti.sh MODEL_DIR -- <training-command...>
# MODEL_DIR holds build_embedded.sh (e.g. build/GpuHostSystemAxi).
# <training-command> runs the representative QEMU workload with
# LLVM_PROFILE_FILE set to "$FLASHSIM_PGO_DIR/%m.profraw".
# The profile is tied to the exact RTL and workload; keep the profile
# outside the source tree and record the trace-build manifest alongside it.
MODEL_DIR="${1:?MODEL_DIR}"
shift
if [ "${1:-}" != "--" ]; then echo "usage: $0 MODEL_DIR -- <training-command...>" >&2; exit 2; fi
shift
TRAINING=("$@")
if [ "${#TRAINING[@]}" -eq 0 ]; then echo "no training command after --" >&2; exit 2; fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
PGO_DIR="${FLASHSIM_PGO_DIR:-$(mktemp -d /tmp/pgo_arti.XXXXXX)}"
mkdir -p "$PGO_DIR"
echo "build instrumented model in $MODEL_DIR"
FLASHSIM_PGO_GENERATE=1 "$MODEL_DIR/build_embedded.sh"
echo "training (LLVM_PROFILE_FILE=$PGO_DIR/%m.profraw)"
if [ -e "$PGO_DIR/merged.profdata" ]; then rm -f "$PGO_DIR/merged.profdata"; fi
rm -f "$PGO_DIR"/*.profraw
LLVM_PROFILE_FILE="$PGO_DIR/%m.profraw" "${TRAINING[@]}"
ls -lh "$PGO_DIR"/*.profraw
MERGE_BIN="$(command -v llvm-profdata || true)"
if [ -z "$MERGE_BIN" ]; then MERGE_BIN="xcrun llvm-profdata"; fi
# shellcheck disable=SC2086
$MERGE_BIN merge -output="$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw
ls -lh "$PGO_DIR/merged.profdata"
echo "build PGO model from $PGO_DIR/merged.profdata"
FLASHSIM_PGO_PROFILE="$PGO_DIR/merged.profdata" "$MODEL_DIR/build_embedded.sh"
echo "done; profile kept at $PGO_DIR/merged.profdata"
