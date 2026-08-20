#!/bin/bash
# Run every scripts/audit-*.mjs against the running dashboard and report which ones still hold.
#
# Each audit is a claim about the RENDERED page, so they rot differently from unit tests: a renamed
# class, reworded copy or a moved control breaks the audit while every test stays green. Nothing ran
# them together before, and 17 hand-run scripts are 17 scripts nobody runs.
#
# THE DETECTOR IS THE EXIT CODE, deliberately. The audits report success in their own wording
# ("calibrate command: the page names the id every arm actually loads"), so grepping for a word like
# PASS marks healthy audits as broken - which is exactly what my first sweep did, and a sweep that
# lies about its own subjects is worse than no sweep. exit 0 = held, non-zero = the page changed under
# the audit, and its log is printed.
#
# Read-only against the live rig: every audit either stubs its API/websocket or only reads, so nothing
# is spawned, recorded, moved or written. Safe with the arms powered.
#
# Usage: scripts/audit-all.sh [name-filter]     e.g. scripts/audit-all.sh devices
set -u
cd "$(dirname "$0")/.."
FILTER="${1:-}"
OUT="${AUDIT_SWEEP_DIR:-/tmp/audit-sweep}"
JOBS="${AUDIT_JOBS:-4}"      # 4 chromium instances at once: more just makes each one slower
rm -rf "$OUT"; mkdir -p "$OUT"

list=$(ls scripts/audit-*.mjs | { [ -n "$FILTER" ] && grep -- "$FILTER" || cat; })
[ -z "$list" ] && { echo "no audits match '$FILTER'"; exit 2; }
echo "$list" | wc -l | xargs printf "running %s audit(s), %s at a time\n" - >/dev/null
printf 'running %s audit(s), %s at a time\n\n' "$(echo "$list" | wc -l | tr -d ' ')" "$JOBS"

# Plain job control, not xargs: `xargs -I{} sh -c '<script>'` refuses this body on macOS with
# "command line cannot be assembled, too long", and it did so silently enough to report 1 broke / 0 held.
run_one() {
  n=$(basename "$1" .mjs); start=$(date +%s)
  node "$1" > "$OUT/$n.log" 2>&1
  echo "$?" > "$OUT/$n.exit"
  echo "$(( $(date +%s) - start ))" > "$OUT/$n.secs"
}
for f in $list; do
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$JOBS" ]; do sleep 1; done
  run_one "$f" &
done
wait

held=0; broke=0
for e in "$OUT"/*.exit; do
  n=$(basename "$e" .exit); code=$(cat "$e"); secs=$(cat "$OUT/$n.secs" 2>/dev/null)
  if [ "$code" = "0" ]; then held=$((held+1)); printf '  held   %-42s %ss\n' "$n" "$secs"
  else broke=$((broke+1)); printf '  BROKE  %-42s %ss (exit %s)\n' "$n" "$secs" "$code"; fi
done
printf '\n%s held, %s broke\n' "$held" "$broke"
if [ "$broke" -gt 0 ]; then
  for e in "$OUT"/*.exit; do
    n=$(basename "$e" .exit)
    [ "$(cat "$e")" = "0" ] || { printf '\n----- %s -----\n' "$n"; tail -14 "$OUT/$n.log"; }
  done
  exit 1
fi
