#!/usr/bin/env bash
# A/B wall-clock of FlashSim vs Verilator QEMU on the OpenGPU probe path
# (real host MMIO + job-queue self-test — the business settle/IRQ path).
#
# Usage:
#   ./scripts/bench_arti_backends.sh
#   FLASHSIM_QEMU=... VERILATOR_QEMU=... ROUNDS=3 ./scripts/bench_arti_backends.sh
set -euo pipefail
# Ensure pipeline failures (e.g. qemu killed) abort the script.

FLASH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ARTI_WORK="${ARTI_WORK:-$(cd "$FLASH_DIR/.." && pwd)/arti-work}"
WORK="${WORK:-/tmp/flashsim-probe-timing}"
KERNEL="${KERNEL:-$ARTI_WORK/arti-linux-build/arch/arm64/boot/Image}"
INITRD="${INITRD:-$WORK/initramfs.cpio.gz}"
FW="${QEMU_FW_DIR:-$ARTI_WORK/qemu-pc-bios}"
FLASHSIM_QEMU="${FLASHSIM_QEMU:-$ARTI_WORK/qemu-arti-build/qemu-system-aarch64}"
VERILATOR_QEMU="${VERILATOR_QEMU:-$ARTI_WORK/qemu-arti-build/qemu-system-aarch64.verilator}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
ROUNDS="${ROUNDS:-3}"
TIMEOUT_BIN="$(command -v gtimeout || command -v timeout)"

fail() { echo "FAIL: $*" >&2; exit 1; }
[ -x "$TIMEOUT_BIN" ] || fail "timeout/gtimeout not found"
[ -f "$KERNEL" ] || fail "kernel missing: $KERNEL"
[ -f "$INITRD" ] || fail "initrd missing: $INITRD (build probe initramfs first)"
[ -x "$FLASHSIM_QEMU" ] || fail "FlashSim QEMU missing: $FLASHSIM_QEMU"
[ -x "$VERILATOR_QEMU" ] || fail "Verilator QEMU missing: $VERILATOR_QEMU"

backend_of() {
  local syms
  syms="$(nm "$1" 2>/dev/null || true)"
  case "$syms" in
    *GpuHostSystemAxiDut*) echo flashsim ;;
    *VGpuHostSystemAxi*) echo verilator ;;
    *) echo unknown ;;
  esac
}

[ "$(backend_of "$FLASHSIM_QEMU")" = flashsim ] || \
  fail "$FLASHSIM_QEMU is not FlashSim-linked"
[ "$(backend_of "$VERILATOR_QEMU")" = verilator ] || \
  fail "$VERILATOR_QEMU is not Verilator-linked"

# Prints one result line; sets LAST_WALL (seconds) for the caller.
run_one() {
  local name="$1" qemu="$2" serial wall rc ready pass
  serial="$(mktemp -t arti-bench-XXXXXX.log)"
  local start end
  start=$(date +%s)
  set +e
  env ${ARTI_MODEL_STATS:+ARTI_MODEL_STATS=$ARTI_MODEL_STATS} \
    "$TIMEOUT_BIN" "$TIMEOUT_SEC" "$qemu" \
    -L "$FW" \
    -machine virt -cpu cortex-a53 -m 1G -smp 2 \
    -nographic -serial mon:stdio -monitor none -display none \
    -kernel "$KERNEL" -initrd "$INITRD" \
    -append "console=ttyAMA0" \
    </dev/null >"$serial" 2>&1
  rc=$?
  set -e
  end=$(date +%s)
  wall=$((end - start))
  LAST_WALL=$wall
  ready=$(grep -E 'job queue ready' "$serial" | head -1 | sed -E 's/.*\[ *([0-9.]+)\].*/\1/' || true)
  pass=$(grep -E 'OPENGPU DRIVER PASS' "$serial" | head -1 | sed -E 's/.*\[ *([0-9.]+)\].*/\1/' || true)
  local ok=0
  grep -q 'PROBE_TIMING: driver load returned to init' "$serial" && ok=1
  local self=NA
  if [ -n "${ready:-}" ] && [ -n "${pass:-}" ]; then
    self=$(python3 -c "print(round(float('$pass')-float('$ready'), 3))")
  fi
  echo "$name wall=${wall}s selftest_guest=${self}s rc=$rc ok=$ok"
  # Keep serial on failure for diagnosis
  if [ "$ok" = 1 ]; then
    rm -f "$serial"
  else
    echo "  log: $serial" >&2
  fi
  [ "$ok" = 1 ]
}

echo "=== ARTI backend A/B (probe self-test) rounds=$ROUNDS ==="
echo "FlashSim : $FLASHSIM_QEMU"
echo "Verilator: $VERILATOR_QEMU"

fs_sum=0 vl_sum=0
LAST_WALL=0
for i in $(seq 1 "$ROUNDS"); do
  echo "--- round $i ---"
  run_one flashsim "$FLASHSIM_QEMU"
  fs_sum=$((fs_sum + LAST_WALL))
  run_one verilator "$VERILATOR_QEMU"
  vl_sum=$((vl_sum + LAST_WALL))
done

python3 - <<PY
fs=$fs_sum; vl=$vl_sum; n=$ROUNDS
print(f"=== mean wall ===")
print(f"flashsim  {fs/n:.1f}s")
print(f"verilator {vl/n:.1f}s")
if vl and fs:
    print(f"speedup   {vl/fs:.2f}x (Verilator_wall / FlashSim_wall; >1 means FlashSim faster)")
PY
